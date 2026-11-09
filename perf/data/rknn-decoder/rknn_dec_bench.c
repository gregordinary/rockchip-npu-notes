// SPDX-License-Identifier: GPL-3.0-or-later
//
// Additional permission under GNU GPL version 3 section 7: if you modify this program, or any
// covered work, by linking or combining it with Rockchip's librknnrt (or a modified version of
// that library), containing parts covered by the terms of Rockchip's license for it, the
// licensors of this program grant you additional permission to convey the resulting work.
//
// rknn_dec_bench: time a whole-step RKNN Whisper decoder (boundarybitlabs/rkwhisper-models) on
// the vendor runtime, the way rkwhisper calls it and with every input resident.
//
//   rknn_dec_bench step <decoder.rknn> <mask: 0|012|auto> <set|res> <warmup> <steps>
//   rknn_dec_bench perf <decoder.rknn> <mask>
//   rknn_dec_bench asr  <mel.rknn> <encoder.rknn> <enc_kv.rknn> <decoder.rknn> <wav> <set|res> <max_new>
//
// The decoder graph (their exporter): inputs token [1,1] int64, past_k/past_v per layer
// [1,H,T,D] fed as native NC1HWC2, enc_k/enc_v per layer [1,S,H,D] as native NC1HWC2,
// self_attn_mask [1,1,1,T], kv_insert_mask and kv_retain_mask [1,1,T,1], pos [1] int64.
// Outputs: logits [1,1,V], present_k/present_v per layer [1,H,1,D].
//
// "set" reproduces rkwhisper's step: every input through rknn_inputs_set (the cross-attention
// K/V of every layer re-sent each step), rknn_run, every output through rknn_outputs_get.
// "res" binds every input and output once with rknn_set_io_mem and writes only the token, the
// position, the masks and the new cache slot per step.
//
// step prints per-step wall split three ways and process CPU time per step (all threads, so a
// runtime worker that spins is counted). asr runs their mel, encoder and cross-K/V models, then
// the decoder greedy, and prints the token ids for scoring against another implementation.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rknn_api.h"

#define MAXIO 256
typedef _Float16 f16;

static double now_ms(clockid_t c) {
    struct timespec t;
    clock_gettime(c, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
#define WALL() now_ms(CLOCK_MONOTONIC)
#define CPU() now_ms(CLOCK_PROCESS_CPUTIME_ID)

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(const double *v, int n) {
    double *w = malloc(n * sizeof *w);
    memcpy(w, v, n * sizeof *w);
    qsort(w, n, sizeof *w, cmp);
    double m = n % 2 ? w[n / 2] : 0.5 * (w[n / 2 - 1] + w[n / 2]);
    free(w);
    return m;
}

static void *slurp(const char *path, long *sz) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *b = malloc(*sz);
    if (fread(b, 1, *sz, f) != (size_t)*sz) { fprintf(stderr, "%s: short read\n", path); exit(1); }
    fclose(f);
    return b;
}

#define CHECK(call)                                                                  \
    do {                                                                             \
        int r_ = (call);                                                             \
        if (r_) { fprintf(stderr, "%s:%d %s = %d\n", __FILE__, __LINE__, #call, r_); exit(1); } \
    } while (0)

static rknn_context load(const char *path, const char *mask_s, uint32_t flags) {
    long sz;
    void *blob = slurp(path, &sz);
    rknn_context ctx;
    CHECK(rknn_init(&ctx, blob, sz, flags, NULL));
    free(blob);
    rknn_core_mask m = !strcmp(mask_s, "012") ? RKNN_NPU_CORE_0_1_2
                     : !strcmp(mask_s, "0")   ? RKNN_NPU_CORE_0
                                              : RKNN_NPU_CORE_AUTO;
    CHECK(rknn_set_core_mask(ctx, m));
    return ctx;
}

static void print_attr(const char *tag, const rknn_tensor_attr *a) {
    printf("  %s[%u] %-18s dims", tag, a->index, a->name);
    for (uint32_t i = 0; i < a->n_dims; i++) printf("%c%u", i ? ',' : '[', a->dims[i]);
    printf("] %s %s n_elems %u size %u stride_size %u w_stride %u\n", get_type_string(a->type),
           get_format_string(a->fmt), a->n_elems, a->size, a->size_with_stride, a->w_stride);
}

/* ---- the decoder --------------------------------------------------------------------------- */

typedef struct {
    rknn_context ctx;
    int res;                     // 1: inputs and outputs bound with rknn_set_io_mem
    int L, H, D, T, S, V;
    uint32_t n_in, n_out;
    rknn_tensor_attr ia[MAXIO], ina[MAXIO], oa[MAXIO], ona[MAXIO];
    // Host-side state in their layouts. In res mode past/enc point into the bound memory.
    f16 **past_k, **past_v, **enc_k, **enc_v;
    f16 *m_attn, *m_ins, *m_ret, *logits, **pres_k, **pres_v;
    int64_t tok, pos;
    rknn_tensor_mem *im[MAXIO], *om[MAXIO];
    int step;
} dec_t;

static size_t nc1hwc2_len(int c, int h, int w) { return (size_t)((c + 7) / 8) * h * w * 8; }

// Indices into the flat input list.
#define I_TOK 0
#define I_PK(d, l) (1 + (l))
#define I_PV(d, l) (1 + (d)->L + (l))
#define I_EK(d, l) (1 + 2 * (d)->L + (l))
#define I_EV(d, l) (1 + 3 * (d)->L + (l))
#define I_MA(d) (4 * (d)->L + 1)
#define I_MI(d) (4 * (d)->L + 2)
#define I_MR(d) (4 * (d)->L + 3)
#define I_POS(d) (4 * (d)->L + 4)

// Write a logical [C][H][W] tensor into a native buffer described by `na`. NC1HWC2 native
// is [1, ceil(C/C2), H, w_stride, C2]; anything else is taken as already contiguous.
static void to_native(const rknn_tensor_attr *na, const f16 *src, int C, int H, int W, f16 *dst) {
    if (na->fmt != RKNN_TENSOR_NC1HWC2) { memcpy(dst, src, (size_t)C * H * W * sizeof *src); return; }
    int c2 = na->dims[4], ws = na->w_stride ? na->w_stride : na->dims[3];
    memset(dst, 0, na->size_with_stride);
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                dst[(((size_t)(c / c2) * H + h) * ws + w) * c2 + c % c2] = src[((size_t)c * H + h) * W + w];
}

static void from_native(const rknn_tensor_attr *na, const f16 *src, int C, int H, int W, f16 *dst) {
    if (na->fmt != RKNN_TENSOR_NC1HWC2) { memcpy(dst, src, (size_t)C * H * W * sizeof *src); return; }
    int c2 = na->dims[4], ws = na->w_stride ? na->w_stride : na->dims[3];
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                dst[((size_t)c * H + h) * W + w] = src[(((size_t)(c / c2) * H + h) * ws + w) * c2 + c % c2];
}

static void dec_open(dec_t *d, const char *path, const char *mask_s, int res, uint32_t flags) {
    memset(d, 0, sizeof *d);
    d->ctx = load(path, mask_s, flags);
    d->res = res;
    rknn_input_output_num io;
    CHECK(rknn_query(d->ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof io));
    d->n_in = io.n_input;
    d->n_out = io.n_output;
    if (d->n_in < 9 || (d->n_in - 5) % 4 || d->n_in > MAXIO || d->n_out > MAXIO) {
        fprintf(stderr, "unexpected io count in %u out %u\n", d->n_in, d->n_out);
        exit(1);
    }
    d->L = (d->n_in - 5) / 4;
    if (d->n_out != (uint32_t)(2 * d->L + 1)) { fprintf(stderr, "out %u != 2L+1\n", d->n_out); exit(1); }
    printf("decoder %s: in %u out %u L %d\n", path, d->n_in, d->n_out, d->L);
    for (uint32_t i = 0; i < d->n_in; i++) {
        d->ia[i].index = d->ina[i].index = i;
        CHECK(rknn_query(d->ctx, RKNN_QUERY_INPUT_ATTR, &d->ia[i], sizeof d->ia[i]));
        CHECK(rknn_query(d->ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &d->ina[i], sizeof d->ina[i]));
        if (i <= 1 || i == (uint32_t)I_EK(d, 0) || i >= (uint32_t)I_MA(d)) {
            print_attr("in ", &d->ia[i]);
            print_attr("inN", &d->ina[i]);
        }
    }
    for (uint32_t i = 0; i < d->n_out; i++) {
        d->oa[i].index = d->ona[i].index = i;
        CHECK(rknn_query(d->ctx, RKNN_QUERY_OUTPUT_ATTR, &d->oa[i], sizeof d->oa[i]));
        CHECK(rknn_query(d->ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &d->ona[i], sizeof d->ona[i]));
        if (i <= 1) { print_attr("out", &d->oa[i]); print_attr("outN", &d->ona[i]); }
    }
    // Geometry from the model, not from a table: T from the mask, D from the native cache's
    // innermost dimension, H from the present output, S from the cross-K input, V from the logits.
    d->T = d->ia[I_MA(d)].n_elems;
    d->D = d->ina[I_PK(d, 0)].dims[3];
    d->H = d->oa[1].n_elems / d->D;
    d->S = d->ia[I_EK(d, 0)].n_elems / (d->H * d->D);
    d->V = d->oa[0].n_elems;
    printf("geometry: L %d H %d D %d T %d S %d V %d\n", d->L, d->H, d->D, d->T, d->S, d->V);
    size_t pl = nc1hwc2_len(d->H, d->T, d->D), el = nc1hwc2_len(d->S, d->H, d->D);
    if (d->ina[I_PK(d, 0)].size_with_stride != pl * 2 || d->ina[I_EK(d, 0)].size_with_stride != el * 2)
        printf("WARNING native sizes %u / %u differ from their NC1HWC2 lengths %zu / %zu\n",
               d->ina[I_PK(d, 0)].size_with_stride, d->ina[I_EK(d, 0)].size_with_stride, pl * 2, el * 2);

    d->past_k = calloc(d->L, sizeof(f16 *));
    d->past_v = calloc(d->L, sizeof(f16 *));
    d->enc_k = calloc(d->L, sizeof(f16 *));
    d->enc_v = calloc(d->L, sizeof(f16 *));
    d->pres_k = calloc(d->L, sizeof(f16 *));
    d->pres_v = calloc(d->L, sizeof(f16 *));
    d->m_attn = calloc(d->T, sizeof(f16));
    d->m_ins = calloc(d->T, sizeof(f16));
    d->m_ret = calloc(d->T, sizeof(f16));
    d->logits = calloc(d->V, sizeof(f16));
    for (int l = 0; l < d->L; l++) {
        d->pres_k[l] = calloc(d->H * d->D, sizeof(f16));
        d->pres_v[l] = calloc(d->H * d->D, sizeof(f16));
    }
    if (!res) {
        for (int l = 0; l < d->L; l++) {
            d->past_k[l] = calloc(pl, sizeof(f16));
            d->past_v[l] = calloc(pl, sizeof(f16));
            d->enc_k[l] = calloc(el, sizeof(f16));
            d->enc_v[l] = calloc(el, sizeof(f16));
        }
        return;
    }
    // Resident: one buffer per tensor in its native form, bound once.
    for (uint32_t i = 0; i < d->n_in; i++) {
        d->im[i] = rknn_create_mem(d->ctx, d->ina[i].size_with_stride);
        if (!d->im[i]) { fprintf(stderr, "create_mem in %u\n", i); exit(1); }
        memset(d->im[i]->virt_addr, 0, d->ina[i].size_with_stride);
        rknn_tensor_attr a = d->ina[i];
        a.pass_through = 1;
        int r = rknn_set_io_mem(d->ctx, d->im[i], &a);
        if (r) { fprintf(stderr, "set_io_mem in %u (%s %s): %d\n", i, get_type_string(a.type), get_format_string(a.fmt), r); exit(1); }
    }
    for (uint32_t i = 0; i < d->n_out; i++) {
        d->om[i] = rknn_create_mem(d->ctx, d->ona[i].size_with_stride);
        if (!d->om[i]) { fprintf(stderr, "create_mem out %u\n", i); exit(1); }
        int r = rknn_set_io_mem(d->ctx, d->om[i], &d->ona[i]);
        if (r) { fprintf(stderr, "set_io_mem out %u: %d\n", i, r); exit(1); }
    }
    for (int l = 0; l < d->L; l++) {
        d->past_k[l] = d->im[I_PK(d, l)]->virt_addr;
        d->past_v[l] = d->im[I_PV(d, l)]->virt_addr;
        d->enc_k[l] = d->im[I_EK(d, l)]->virt_addr;
        d->enc_v[l] = d->im[I_EV(d, l)]->virt_addr;
    }
    for (uint32_t i = 0; i < d->n_in; i++) CHECK(rknn_mem_sync(d->ctx, d->im[i], RKNN_MEMORY_SYNC_TO_DEVICE));
}

// Their pack: logical [S][H][D] into native [ceil(S/8)][H][D][8].
static void pack_enc(const f16 *src, f16 *dst, int s, int h, int dd) {
    memset(dst, 0, nc1hwc2_len(s, h, dd) * sizeof *dst);
    for (int q = 0; q < s; q++)
        for (int i = 0; i < h; i++)
            for (int j = 0; j < dd; j++)
                dst[(((size_t)(q / 8) * h + i) * dd + j) * 8 + q % 8] = src[((size_t)q * h + i) * dd + j];
}

static void dec_set_enc(dec_t *d, f16 **ek, f16 **ev) {
    for (int l = 0; l < d->L; l++) {
        pack_enc(ek[l], d->enc_k[l], d->S, d->H, d->D);
        pack_enc(ev[l], d->enc_v[l], d->S, d->H, d->D);
        if (d->res) {
            CHECK(rknn_mem_sync(d->ctx, d->im[I_EK(d, l)], RKNN_MEMORY_SYNC_TO_DEVICE));
            CHECK(rknn_mem_sync(d->ctx, d->im[I_EV(d, l)], RKNN_MEMORY_SYNC_TO_DEVICE));
        }
    }
}

static void dec_reset(dec_t *d) {
    size_t pl = nc1hwc2_len(d->H, d->T, d->D);
    for (int l = 0; l < d->L; l++) {
        memset(d->past_k[l], 0, pl * sizeof(f16));
        memset(d->past_v[l], 0, pl * sizeof(f16));
        if (d->res) {
            CHECK(rknn_mem_sync(d->ctx, d->im[I_PK(d, l)], RKNN_MEMORY_SYNC_TO_DEVICE));
            CHECK(rknn_mem_sync(d->ctx, d->im[I_PV(d, l)], RKNN_MEMORY_SYNC_TO_DEVICE));
        }
    }
    d->step = 0;
}

typedef struct { double set, run, get, cpu; } step_t;

// One decoder step, their arithmetic: masks, run, splice the new K/V into the cache at
// pos % T. Logits land in d->logits as fp16.
static step_t dec_step(dec_t *d, int token) {
    step_t st;
    double c0 = CPU(), a = WALL();
    int T = d->T, pos = d->step, slot = pos % T;
    for (int i = 0; i < T; i++) {
        d->m_attn[i] = (pos >= T || i <= pos) ? (f16)0 : (f16)-65504;
        d->m_ins[i] = i == slot ? (f16)1 : (f16)0;
        d->m_ret[i] = (i != slot && (pos >= T || i < pos)) ? (f16)1 : (f16)0;
    }
    d->tok = token;
    d->pos = pos;
    if (!d->res) {
        rknn_input in[MAXIO];
        memset(in, 0, sizeof in);
        in[I_TOK] = (rknn_input){.index = I_TOK, .buf = &d->tok, .size = 8, .type = RKNN_TENSOR_INT64,
                                 .fmt = RKNN_TENSOR_UNDEFINED};
        size_t pl = nc1hwc2_len(d->H, T, d->D) * 2, el = nc1hwc2_len(d->S, d->H, d->D) * 2;
        for (int l = 0; l < d->L; l++) {
            in[I_PK(d, l)] = (rknn_input){.index = I_PK(d, l), .buf = d->past_k[l], .size = pl,
                                          .pass_through = 1, .type = RKNN_TENSOR_FLOAT16, .fmt = RKNN_TENSOR_NC1HWC2};
            in[I_PV(d, l)] = (rknn_input){.index = I_PV(d, l), .buf = d->past_v[l], .size = pl,
                                          .pass_through = 1, .type = RKNN_TENSOR_FLOAT16, .fmt = RKNN_TENSOR_NC1HWC2};
            in[I_EK(d, l)] = (rknn_input){.index = I_EK(d, l), .buf = d->enc_k[l], .size = el,
                                          .pass_through = 1, .type = RKNN_TENSOR_FLOAT16, .fmt = RKNN_TENSOR_NC1HWC2};
            in[I_EV(d, l)] = (rknn_input){.index = I_EV(d, l), .buf = d->enc_v[l], .size = el,
                                          .pass_through = 1, .type = RKNN_TENSOR_FLOAT16, .fmt = RKNN_TENSOR_NC1HWC2};
        }
        in[I_MA(d)] = (rknn_input){.index = I_MA(d), .buf = d->m_attn, .size = T * 2, .type = RKNN_TENSOR_FLOAT16,
                                   .fmt = RKNN_TENSOR_NHWC};
        in[I_MI(d)] = (rknn_input){.index = I_MI(d), .buf = d->m_ins, .size = T * 2, .type = RKNN_TENSOR_FLOAT16,
                                   .fmt = RKNN_TENSOR_NHWC};
        in[I_MR(d)] = (rknn_input){.index = I_MR(d), .buf = d->m_ret, .size = T * 2, .type = RKNN_TENSOR_FLOAT16,
                                   .fmt = RKNN_TENSOR_NHWC};
        in[I_POS(d)] = (rknn_input){.index = I_POS(d), .buf = &d->pos, .size = 8, .type = RKNN_TENSOR_INT64,
                                    .fmt = RKNN_TENSOR_UNDEFINED};
        CHECK(rknn_inputs_set(d->ctx, d->n_in, in));
    } else {
        memcpy(d->im[I_TOK]->virt_addr, &d->tok, 8);
        memcpy(d->im[I_POS(d)]->virt_addr, &d->pos, 8);
        // self_attn_mask is [1,1,1,T] (C=1, H=1, W=T); the insert and retain masks [1,1,T,1].
        to_native(&d->ina[I_MA(d)], d->m_attn, 1, 1, T, d->im[I_MA(d)]->virt_addr);
        to_native(&d->ina[I_MI(d)], d->m_ins, 1, T, 1, d->im[I_MI(d)]->virt_addr);
        to_native(&d->ina[I_MR(d)], d->m_ret, 1, T, 1, d->im[I_MR(d)]->virt_addr);
        int sync[] = {I_TOK, I_POS(d), I_MA(d), I_MI(d), I_MR(d)};
        for (int k = 0; k < 5; k++) CHECK(rknn_mem_sync(d->ctx, d->im[sync[k]], RKNN_MEMORY_SYNC_TO_DEVICE));
    }
    double b = WALL();
    CHECK(rknn_run(d->ctx, NULL));
    double c = WALL();
    if (!d->res) {
        rknn_output out[MAXIO];
        memset(out, 0, sizeof out);
        out[0] = (rknn_output){.index = 0, .want_float = 0, .is_prealloc = 1, .buf = d->logits, .size = d->V * 2};
        for (int l = 0; l < d->L; l++) {
            out[1 + l] = (rknn_output){.index = 1 + l, .is_prealloc = 1, .buf = d->pres_k[l], .size = d->H * d->D * 2};
            out[1 + d->L + l] = (rknn_output){.index = 1 + d->L + l, .is_prealloc = 1, .buf = d->pres_v[l],
                                              .size = d->H * d->D * 2};
        }
        CHECK(rknn_outputs_get(d->ctx, d->n_out, out, NULL));
        CHECK(rknn_outputs_release(d->ctx, d->n_out, out));
    } else {
        for (uint32_t i = 0; i < d->n_out; i++) CHECK(rknn_mem_sync(d->ctx, d->om[i], RKNN_MEMORY_SYNC_FROM_DEVICE));
        from_native(&d->ona[0], d->om[0]->virt_addr, d->V, 1, 1, d->logits);
        for (int l = 0; l < d->L; l++) {
            // present is [1,H,1,D]: C=H, H=1, W=D.
            from_native(&d->ona[1 + l], d->om[1 + l]->virt_addr, d->H, 1, d->D, d->pres_k[l]);
            from_native(&d->ona[1 + d->L + l], d->om[1 + d->L + l]->virt_addr, d->H, 1, d->D, d->pres_v[l]);
        }
    }
    // Their splice: present [H][D] into the cache [ceil(H/8)][T][D][8] at slot.
    for (int l = 0; l < d->L; l++)
        for (int h = 0; h < d->H; h++)
            for (int j = 0; j < d->D; j++) {
                size_t o = (((size_t)(h / 8) * T + slot) * d->D + j) * 8 + h % 8;
                d->past_k[l][o] = d->pres_k[l][h * d->D + j];
                d->past_v[l][o] = d->pres_v[l][h * d->D + j];
            }
    if (d->res)
        for (int l = 0; l < d->L; l++) {
            CHECK(rknn_mem_sync(d->ctx, d->im[I_PK(d, l)], RKNN_MEMORY_SYNC_TO_DEVICE));
            CHECK(rknn_mem_sync(d->ctx, d->im[I_PV(d, l)], RKNN_MEMORY_SYNC_TO_DEVICE));
        }
    d->step++;
    double e = WALL();
    st.set = b - a;
    st.run = c - b;
    st.get = e - c;
    st.cpu = CPU() - c0;
    return st;
}

static void logits_stats(const dec_t *d) {
    size_t bad = 0;
    double sum = 0, sq = 0;
    int am = 0;
    for (int i = 0; i < d->V; i++) {
        float v = d->logits[i];
        if (!isfinite(v)) { bad++; continue; }
        sum += v;
        sq += (double)v * v;
        if (v > (float)d->logits[am]) am = i;
    }
    double mean = sum / d->V;
    printf("logits: non-finite %zu mean %.4f std %.4f argmax %d (%.3f)\n", bad, mean,
           sqrt(sq / d->V - mean * mean), am, (float)d->logits[am]);
}

static void report(const char *tag, const step_t *s, int n) {
    double *v = malloc(n * sizeof *v), tot[4] = {0};
    const char *names[] = {"set", "run", "get", "cpu"};
    printf("RESULT %s steps %d", tag, n);
    for (int k = 0; k < 4; k++) {
        for (int i = 0; i < n; i++) {
            v[i] = k == 0 ? s[i].set : k == 1 ? s[i].run : k == 2 ? s[i].get : s[i].cpu;
            tot[k] += v[i];
        }
        printf(" %s_med %.3f", names[k], median(v, n));
    }
    for (int i = 0; i < n; i++) v[i] = s[i].set + s[i].run + s[i].get;
    printf(" step_med %.3f step_mean %.3f cpu_mean %.3f\n", median(v, n), (tot[0] + tot[1] + tot[2]) / n, tot[3] / n);
    free(v);
}

// Deterministic, attention-shaped stand-in for the cross-attention K/V: uniform in [-1, 1].
static void fill_enc_random(dec_t *d) {
    size_t n = (size_t)d->S * d->H * d->D;
    f16 **ek = calloc(d->L, sizeof *ek), **ev = calloc(d->L, sizeof *ev);
    unsigned s = 12345;
    for (int l = 0; l < d->L; l++) {
        ek[l] = malloc(n * sizeof(f16));
        ev[l] = malloc(n * sizeof(f16));
        for (size_t i = 0; i < n; i++) {
            s = s * 1103515245u + 12345u;
            ek[l][i] = (f16)(((s >> 8) & 0xffff) / 32767.5f - 1.0f);
            s = s * 1103515245u + 12345u;
            ev[l][i] = (f16)(((s >> 8) & 0xffff) / 32767.5f - 1.0f);
        }
    }
    dec_set_enc(d, ek, ev);
    for (int l = 0; l < d->L; l++) { free(ek[l]); free(ev[l]); }
    free(ek);
    free(ev);
}

static int cmd_step(int argc, char **argv) {
    if (argc != 7) return 2;
    int res = !strcmp(argv[4], "res"), warm = atoi(argv[5]), n = atoi(argv[6]);
    dec_t d;
    dec_open(&d, argv[2], argv[3], res, 0);
    fill_enc_random(&d);
    dec_reset(&d);
    step_t *s = malloc(n * sizeof *s);
    // Tokens cycle through ordinary text ids. Positions advance and wrap at T as theirs do,
    // so a run longer than T exercises the full-window masks too.
    for (int i = -warm; i < n; i++) {
        step_t t = dec_step(&d, 400 + (i + warm) % 1000);
        if (i == 0) logits_stats(&d);
        if (i >= 0) s[i] = t;
    }
    char tag[64];
    snprintf(tag, sizeof tag, "step mode %s mask %s", argv[4], argv[3]);
    report(tag, s, n);
    return 0;
}

static int cmd_perf(int argc, char **argv) {
    if (argc != 4) return 2;
    dec_t d;
    dec_open(&d, argv[2], argv[3], 0, RKNN_FLAG_COLLECT_PERF_MASK);
    fill_enc_random(&d);
    dec_reset(&d);
    for (int i = 0; i < 8; i++) dec_step(&d, 400 + i);
    rknn_perf_run pr;
    CHECK(rknn_query(d.ctx, RKNN_QUERY_PERF_RUN, &pr, sizeof pr));
    printf("perf_run_us %lld\n", (long long)pr.run_duration);
    rknn_perf_detail pd;
    CHECK(rknn_query(d.ctx, RKNN_QUERY_PERF_DETAIL, &pd, sizeof pd));
    fwrite(pd.perf_data, 1, pd.data_len, stdout);
    printf("\n");
    return 0;
}

/* ---- the whole pipeline, for the token ids -------------------------------------------------- */

#define N_SAMPLES 480000
#define N_FRAMES 3000
#define PAD 200
#define GCD 80

static float *read_wav16k(const char *path, int *n) {
    long sz;
    uint8_t *b = slurp(path, &sz);
    if (sz < 12 || memcmp(b, "RIFF", 4) || memcmp(b + 8, "WAVE", 4)) { fprintf(stderr, "not a WAV\n"); exit(1); }
    int ch = 0, rate = 0, bits = 0;
    for (long p = 12; p + 8 <= sz;) {
        uint32_t len = b[p + 4] | b[p + 5] << 8 | b[p + 6] << 16 | (uint32_t)b[p + 7] << 24;
        if (!memcmp(b + p, "fmt ", 4)) {
            ch = b[p + 10] | b[p + 11] << 8;
            rate = b[p + 12] | b[p + 13] << 8 | b[p + 14] << 16;
            bits = b[p + 22] | b[p + 23] << 8;
        } else if (!memcmp(b + p, "data", 4)) {
            if (ch != 1 || rate != 16000 || bits != 16) {
                fprintf(stderr, "need 16 kHz mono s16, got %d ch %d Hz %d bit\n", ch, rate, bits);
                exit(1);
            }
            *n = len / 2;
            float *x = malloc(*n * sizeof *x);
            for (int i = 0; i < *n; i++) x[i] = (int16_t)(b[p + 8 + 2 * i] | b[p + 9 + 2 * i] << 8) / 32768.0f;
            free(b);
            return x;
        }
        p += 8 + len + (len & 1);
    }
    fprintf(stderr, "no data chunk\n");
    exit(1);
}

// Their polyphase pre-process: reflect-pad 200 each side, pad to a multiple of 80, then
// sample 80t + c to channel c, column t.
static float *polyphase(const float *in, int n, int *cols) {
    int len = PAD + n + PAD;
    len += (GCD - len % GCD) % GCD;
    float *xp = calloc(len, sizeof *xp);
    for (int i = PAD; i >= 1; i--) xp[PAD - i] = in[i < n ? i : n - 1];
    memcpy(xp + PAD, in, n * sizeof *in);
    for (int i = 0; i < PAD; i++) xp[PAD + n + i] = in[n > 1 + i ? n - 2 - i : 0];
    *cols = len / GCD;
    float *out = malloc(len * sizeof *out);
    for (int c = 0; c < GCD; c++)
        for (int t = 0; t < *cols; t++) out[c * *cols + t] = xp[t * GCD + c];
    free(xp);
    return out;
}

static int cmd_asr(int argc, char **argv) {
    if (argc != 9) return 2;
    int res = !strcmp(argv[7], "res"), max_new = atoi(argv[8]);
    int n_audio;
    float *audio = read_wav16k(argv[6], &n_audio);
    float *wave = calloc(N_SAMPLES, sizeof *wave);
    memcpy(wave, audio, (n_audio < N_SAMPLES ? n_audio : N_SAMPLES) * sizeof *wave);
    int cols;
    float *poly = polyphase(wave, N_SAMPLES, &cols);
    printf("audio %d samples (%.2f s), polyphase [80,%d]\n", n_audio, n_audio / 16000.0, cols);

    // Mel.
    double t0 = WALL();
    rknn_context mel = load(argv[2], "0", 0);
    rknn_tensor_attr a = {.index = 0};
    CHECK(rknn_query(mel, RKNN_QUERY_INPUT_ATTR, &a, sizeof a));
    print_attr("mel in ", &a);
    if (a.n_elems != (uint32_t)(GCD * cols)) { fprintf(stderr, "mel input %u != %d\n", a.n_elems, GCD * cols); exit(1); }
    rknn_input mi = {.index = 0, .buf = poly, .size = GCD * cols * 4, .type = RKNN_TENSOR_FLOAT32, .fmt = a.fmt};
    float *mels = malloc(80 * N_FRAMES * sizeof *mels);
    rknn_output mo = {.index = 0, .want_float = 1, .is_prealloc = 1, .buf = mels, .size = 80 * N_FRAMES * 4};
    double t1 = WALL();
    CHECK(rknn_inputs_set(mel, 1, &mi));
    CHECK(rknn_run(mel, NULL));
    CHECK(rknn_outputs_get(mel, 1, &mo, NULL));
    double t2 = WALL();
    float mx = -INFINITY;
    for (int i = 0; i < 80 * N_FRAMES; i++) {
        mels[i] = log10f(fmaxf(mels[i], 1e-10f));
        if (mels[i] > mx) mx = mels[i];
    }
    double msum = 0;
    for (int i = 0; i < 80 * N_FRAMES; i++) {
        mels[i] = (fmaxf(mels[i], mx - 8.0f) + 4.0f) / 4.0f;
        msum += mels[i];
    }
    printf("mel: %.2f ms run, mean %.4f max-log %.4f\n", t2 - t1, msum / (80 * N_FRAMES), mx);

    // Encoder.
    rknn_context enc = load(argv[3], "0", 0);
    rknn_tensor_attr eo = {.index = 0};
    CHECK(rknn_query(enc, RKNN_QUERY_OUTPUT_ATTR, &eo, sizeof eo));
    print_attr("enc out", &eo);
    rknn_input ei = {.index = 0, .buf = mels, .size = 80 * N_FRAMES * 4, .type = RKNN_TENSOR_FLOAT32,
                     .fmt = RKNN_TENSOR_UNDEFINED};
    f16 *hid = malloc(eo.n_elems * sizeof *hid);
    rknn_output eout = {.index = 0, .want_float = 0, .is_prealloc = 1, .buf = hid, .size = eo.n_elems * 2};
    t1 = WALL();
    CHECK(rknn_inputs_set(enc, 1, &ei));
    CHECK(rknn_run(enc, NULL));
    CHECK(rknn_outputs_get(enc, 1, &eout, NULL));
    t2 = WALL();
    printf("encoder: %.2f ms\n", t2 - t1);

    // Decoder geometry first, so the cross-K/V buffers are sized from it.
    dec_t d;
    dec_open(&d, argv[5], "0", res, 0);

    // Cross K/V: outputs interleaved k_l0, v_l0, k_l1, ...
    rknn_context kv = load(argv[4], "0", 0);
    rknn_input ki = {.index = 0, .buf = hid, .size = eo.n_elems * 2, .type = RKNN_TENSOR_FLOAT16,
                     .fmt = RKNN_TENSOR_UNDEFINED};
    size_t pl = (size_t)d.S * d.H * d.D;
    f16 **ek = calloc(d.L, sizeof *ek), **ev = calloc(d.L, sizeof *ev);
    rknn_output ko[MAXIO];
    memset(ko, 0, sizeof ko);
    for (int l = 0; l < d.L; l++) {
        ek[l] = malloc(pl * sizeof(f16));
        ev[l] = malloc(pl * sizeof(f16));
        ko[2 * l] = (rknn_output){.index = 2 * l, .is_prealloc = 1, .buf = ek[l], .size = pl * 2};
        ko[2 * l + 1] = (rknn_output){.index = 2 * l + 1, .is_prealloc = 1, .buf = ev[l], .size = pl * 2};
    }
    t1 = WALL();
    CHECK(rknn_inputs_set(kv, 1, &ki));
    CHECK(rknn_run(kv, NULL));
    CHECK(rknn_outputs_get(kv, 2 * d.L, ko, NULL));
    t2 = WALL();
    printf("cross-kv: %.2f ms\n", t2 - t1);
    dec_set_enc(&d, ek, ev);
    dec_reset(&d);
    printf("front end total %.1f ms (load included)\n", WALL() - t0);

    // Greedy, notimestamps. Suppress every special token but end-of-text, and suppress
    // end-of-text and the blank at the first sampled position.
    const int prompt[] = {50258, 50259, 50359, 50363}, EOT = 50257;
    int np = sizeof prompt / sizeof *prompt;
    step_t *s = malloc((np + max_new) * sizeof *s);
    int ns = 0, gen[1024], ng = 0;
    for (int i = 0; i < np; i++) s[ns++] = dec_step(&d, prompt[i]);
    for (int k = 0; k < max_new && ng < 1024; k++) {
        int best = -1;
        float bv = -INFINITY;
        for (int i = 0; i < d.V; i++) {
            if (i > EOT) continue;
            if (ng == 0 && (i == EOT || i == 220)) continue;
            float v = d.logits[i];
            if (v > bv) { bv = v; best = i; }
        }
        if (best == EOT) break;
        gen[ng++] = best;
        if (ng >= 8 && !memcmp(gen + ng - 8, gen + ng - 4, 4 * sizeof *gen)) break;
        s[ns++] = dec_step(&d, best);
    }
    printf("TOKENS");
    for (int i = 0; i < ng; i++) printf(" %d", gen[i]);
    printf("\n");
    char tag[64];
    snprintf(tag, sizeof tag, "asr mode %s", argv[7]);
    report(tag, s, ns);
    return 0;
}

int main(int argc, char **argv) {
    int r = 2;
    if (argc >= 2 && !strcmp(argv[1], "step")) r = cmd_step(argc, argv);
    else if (argc >= 2 && !strcmp(argv[1], "perf")) r = cmd_perf(argc, argv);
    else if (argc >= 2 && !strcmp(argv[1], "asr")) r = cmd_asr(argc, argv);
    if (r == 2)
        fprintf(stderr,
                "usage: %s step <decoder.rknn> <0|012|auto> <set|res> <warmup> <steps>\n"
                "       %s perf <decoder.rknn> <0|012|auto>\n"
                "       %s asr  <mel> <encoder> <enc_kv> <decoder> <wav> <set|res> <max_new>\n",
                argv[0], argv[0], argv[0]);
    return r;
}
