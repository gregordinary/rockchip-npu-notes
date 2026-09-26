// SPDX-License-Identifier: GPL-3.0-or-later
//
// Additional permission under GNU GPL version 3 section 7: if you modify this program, or any
// covered work, by linking or combining it with Rockchip's librknnrt (or a modified version of
// that library), containing parts covered by the terms of Rockchip's license for it, the
// licensors of this program grant you additional permission to convey the resulting work.
//
// rknn_enc_bench: time a whole-graph RKNN encoder on the vendor runtime.
//   rknn_enc_bench <model.rknn> <mask: 012|0|auto> <warmup> <reps>
// Prints per-rep rknn_run time, the set+run+get time, and output statistics from the first
// rep, so an encoder that returns a constant or non-finite surface cannot pass as a fast one.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rknn_api.h"

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n) {
    qsort(v, n, sizeof *v, cmp);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s model.rknn 012|0|auto warmup reps\n", argv[0]);
        return 2;
    }
    const char *path = argv[1], *mask_s = argv[2];
    int warm = atoi(argv[3]), reps = atoi(argv[4]);

    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *blob = malloc(sz);
    if (fread(blob, 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);

    rknn_context ctx;
    int ret = rknn_init(&ctx, blob, sz, 0, NULL);
    if (ret) { fprintf(stderr, "rknn_init: %d\n", ret); return 1; }

    rknn_core_mask mask = !strcmp(mask_s, "012") ? RKNN_NPU_CORE_0_1_2
                        : !strcmp(mask_s, "0")   ? RKNN_NPU_CORE_0
                                                 : RKNN_NPU_CORE_AUTO;
    ret = rknn_set_core_mask(ctx, mask);
    if (ret) { fprintf(stderr, "rknn_set_core_mask(%s): %d\n", mask_s, ret); return 1; }

    rknn_sdk_version ver;
    rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof ver);
    rknn_input_output_num io;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof io);
    rknn_tensor_attr ia = {.index = 0}, oa = {.index = 0};
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &ia, sizeof ia);
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &oa, sizeof oa);
    printf("api %s | driver %s | mask %s | in %u out %u\n", ver.api_version, ver.drv_version,
           mask_s, io.n_input, io.n_output);
    printf("input  %s dims [%u,%u,%u] type %s fmt %s n_elems %u\n", ia.name, ia.dims[0], ia.dims[1],
           ia.dims[2], get_type_string(ia.type), get_format_string(ia.fmt), ia.n_elems);
    printf("output %s dims [%u,%u,%u] type %s n_elems %u\n", oa.name, oa.dims[0], oa.dims[1],
           oa.dims[2], get_type_string(oa.type), oa.n_elems);

    // A mel-like input: deterministic, uniform in [-1, 1]. The window is fixed at compile time,
    // so the element count must match the model's, which the query above reports.
    size_t n_in = ia.n_elems;
    float *x = malloc(n_in * sizeof *x);
    unsigned s = 12345;
    for (size_t i = 0; i < n_in; i++) {
        s = s * 1103515245u + 12345u;
        x[i] = ((s >> 8) & 0xffff) / 32767.5f - 1.0f;
    }

    double *t_run = malloc(reps * sizeof *t_run), *t_all = malloc(reps * sizeof *t_all);
    for (int r = -warm; r < reps; r++) {
        rknn_input in = {.index = 0, .type = RKNN_TENSOR_FLOAT32, .size = n_in * sizeof *x,
                         .fmt = ia.fmt, .buf = x, .pass_through = 0};
        rknn_output out = {.index = 0, .want_float = 1};
        double a = now_ms();
        if ((ret = rknn_inputs_set(ctx, 1, &in))) { fprintf(stderr, "inputs_set: %d\n", ret); return 1; }
        double b = now_ms();
        if ((ret = rknn_run(ctx, NULL))) { fprintf(stderr, "rknn_run: %d\n", ret); return 1; }
        double c = now_ms();
        if ((ret = rknn_outputs_get(ctx, 1, &out, NULL))) { fprintf(stderr, "outputs_get: %d\n", ret); return 1; }
        double d = now_ms();
        if (r == 0) {
            const float *y = out.buf;
            size_t n = out.size / sizeof(float), bad = 0;
            double sum = 0, sq = 0;
            for (size_t i = 0; i < n; i++) {
                if (!isfinite(y[i])) { bad++; continue; }
                sum += y[i];
                sq += (double)y[i] * y[i];
            }
            double mean = sum / n;
            printf("output stats: n %zu non-finite %zu mean %.5f std %.5f y[0..3] %.4f %.4f %.4f %.4f\n",
                   n, bad, mean, sqrt(sq / n - mean * mean), y[0], y[1], y[2], y[3]);
        }
        rknn_outputs_release(ctx, 1, &out);
        if (r >= 0) { t_run[r] = c - b; t_all[r] = d - a; }
    }
    printf("run_ms:");
    for (int r = 0; r < reps; r++) printf(" %.2f", t_run[r]);
    printf("\n");
    double mn = t_run[0], mx = t_run[0];
    for (int r = 1; r < reps; r++) { if (t_run[r] < mn) mn = t_run[r]; if (t_run[r] > mx) mx = t_run[r]; }
    double med_run = median(t_run, reps), med_all = median(t_all, reps);
    printf("RESULT mask %s run_median_ms %.2f run_min_ms %.2f run_max_ms %.2f set_run_get_median_ms %.2f reps %d\n",
           mask_s, med_run, mn, mx, med_all, reps);
    rknn_destroy(ctx);
    return 0;
}
