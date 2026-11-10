// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rockchip-npu-notes authors
//
// wdec_cpu: what whisper.cpp's decoder step costs on the CPU, per call, at several thread counts.
//   wdec_cpu <model.bin> <wav> <threads[,threads...]> <reps>
//
// One encode (4 threads), then the greedy token sequence for the clip from <|sot|><|en|>
// <|transcribe|><|notimestamps|>. Each timed replay feeds that same sequence through
// whisper_decode one token at a time (the prompt pass untimed), so every thread count runs the
// same steps at the same positions. The replays rotate the thread order rep by rep inside one
// process; whisper_decode drops the cache from n_past on, so a replay starts clean.
//
// Between calls a log-softmax over the vocabulary stands in for whisper.cpp's sampling, so the
// CPU a waiting OpenMP team spends between steps is counted. Per thread count it prints the
// median call wall, the median step wall (call plus sampling) and the mean process CPU per step,
// which counts every thread.
//
// Build against the same tree's libwhisper, and run with WDEC_BACKEND_DIR set to the directory
// holding its libggml-cpu.so when that tree was built with GGML_BACKEND_DL:
//   gcc -O2 -o wdec_cpu wdec_cpu.c -I$W/include -I$W/ggml/include -L$W/build/bin -lwhisper -lggml
//       -lm -Wl,-rpath,$W/build/bin
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ggml-backend.h"
#include "whisper.h"

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

static double median(double *v, int n) {
    qsort(v, n, sizeof *v, cmp);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static float *read_wav16k(const char *path, int *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(sz);
    if (fread(b, 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    int ch = 0, rate = 0, bits = 0;
    for (long p = 12; p + 8 <= sz;) {
        uint32_t len = b[p + 4] | b[p + 5] << 8 | b[p + 6] << 16 | (uint32_t)b[p + 7] << 24;
        if (!memcmp(b + p, "fmt ", 4)) {
            ch = b[p + 10] | b[p + 11] << 8;
            rate = b[p + 12] | b[p + 13] << 8 | b[p + 14] << 16;
            bits = b[p + 22] | b[p + 23] << 8;
        } else if (!memcmp(b + p, "data", 4)) {
            if (ch != 1 || rate != 16000 || bits != 16) { fprintf(stderr, "need 16 kHz mono s16\n"); exit(1); }
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

// The sampling stand-in: max, sum of exponentials, log, over the whole vocabulary.
static volatile float g_sink;
static void log_softmax(const float *l, int n) {
    float mx = l[0];
    for (int i = 1; i < n; i++) if (l[i] > mx) mx = l[i];
    double s = 0;
    for (int i = 0; i < n; i++) s += expf(l[i] - mx);
    g_sink = mx + (float)log(s);
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s model.bin clip.wav threads[,threads...] reps\n", argv[0]);
        return 2;
    }
    int th[8], nth = 0, reps = atoi(argv[4]);
    for (char *p = strtok(argv[3], ","); p && nth < 8; p = strtok(NULL, ",")) th[nth++] = atoi(p);

    const char *bdir = getenv("WDEC_BACKEND_DIR");
    if (bdir) ggml_backend_load_all_from_path(bdir);
    else ggml_backend_load_all();
    struct whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = false;
    struct whisper_context *ctx = whisper_init_from_file_with_params(argv[1], cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 1; }
    int n;
    float *pcm = read_wav16k(argv[2], &n);
    if (whisper_pcm_to_mel(ctx, pcm, n, 4) || whisper_encode(ctx, 0, 4)) { fprintf(stderr, "encode failed\n"); return 1; }

    const int nv = whisper_n_vocab(ctx);
    const whisper_token eot = whisper_token_eot(ctx);
    whisper_token prompt[4] = { whisper_token_sot(ctx), whisper_token_lang(ctx, whisper_lang_id("en")),
                                whisper_token_transcribe(ctx), whisper_token_not(ctx) };
    // The greedy sequence, at 4 threads.
    whisper_token seq[128];
    int ns = 0;
    if (whisper_decode(ctx, prompt, 4, 0, 4)) return 1;
    const float *lg = whisper_get_logits(ctx) + (size_t)3 * nv;
    while (ns < 128) {
        int best = -1;
        for (int i = 0; i <= eot; i++) {
            if (ns == 0 && (i == eot || i == 220)) continue;
            if (best < 0 || lg[i] > lg[best]) best = i;
        }
        if (best == eot) break;
        seq[ns] = best;
        if (whisper_decode(ctx, &seq[ns], 1, 4 + ns, 4)) return 1;
        ns++;
        lg = whisper_get_logits(ctx);
    }
    printf("tokens %d:", ns);
    for (int i = 0; i < ns; i++) printf("%s", whisper_token_to_str(ctx, seq[i]));
    printf("\n");

    double *call = malloc(ns * sizeof *call), *step = malloc(ns * sizeof *step);
    for (int r = 0; r < reps; r++)
        for (int k = 0; k < nth; k++) {
            int t = th[(k + r) % nth];
            if (whisper_decode(ctx, prompt, 4, 0, t)) return 1;
            double cpu0 = CPU();
            for (int i = 0; i < ns; i++) {
                double w0 = WALL();
                if (whisper_decode(ctx, &seq[i], 1, 4 + i, t)) return 1;
                double w1 = WALL();
                log_softmax(whisper_get_logits(ctx), nv);
                call[i] = w1 - w0;
                step[i] = WALL() - w0;
            }
            double cpu = (CPU() - cpu0) / ns;
            printf("RESULT rep %d threads %d steps %d call_med %.3f step_med %.3f cpu_per_step %.3f\n", r, t, ns,
                   median(call, ns), median(step, ns), cpu);
            fflush(stdout);
        }
    whisper_free(ctx);
    return 0;
}
