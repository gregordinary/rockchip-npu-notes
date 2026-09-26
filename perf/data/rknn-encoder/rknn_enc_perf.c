// SPDX-License-Identifier: GPL-3.0-or-later
//
// Additional permission under GNU GPL version 3 section 7: if you modify this program, or any
// covered work, by linking or combining it with Rockchip's librknnrt (or a modified version of
// that library), containing parts covered by the terms of Rockchip's license for it, the
// licensors of this program grant you additional permission to convey the resulting work.
//
// rknn_enc_perf: the runtime's own per-op profile of an RKNN model, for attribution only.
//   rknn_enc_perf <model.rknn> <mask: 012|0|auto> <warmup>
// Perf collection changes the timing, so nothing here is a scored wall. It prints
// RKNN_QUERY_PERF_RUN for each run and RKNN_QUERY_PERF_DETAIL (per op: target, time) for the last.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_api.h"

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s model.rknn 012|0|auto warmup\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *blob = malloc(sz);
    if (fread(blob, 1, sz, f) != (size_t)sz) return 1;
    fclose(f);

    rknn_context ctx;
    int ret = rknn_init(&ctx, blob, sz, RKNN_FLAG_COLLECT_PERF_MASK, NULL);
    if (ret) { fprintf(stderr, "rknn_init: %d\n", ret); return 1; }
    const char *m = argv[2];
    rknn_set_core_mask(ctx, !strcmp(m, "012") ? RKNN_NPU_CORE_0_1_2
                          : !strcmp(m, "0") ? RKNN_NPU_CORE_0 : RKNN_NPU_CORE_AUTO);
    rknn_tensor_attr ia = {.index = 0};
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &ia, sizeof ia);
    float *x = calloc(ia.n_elems, sizeof *x);
    for (unsigned i = 0; i < ia.n_elems; i++) x[i] = (float)((i * 2654435761u) % 2001) / 1000.0f - 1.0f;

    int n = atoi(argv[3]) + 1;
    for (int r = 0; r < n; r++) {
        rknn_input in = {.index = 0, .type = RKNN_TENSOR_FLOAT32, .size = ia.n_elems * sizeof *x,
                         .fmt = ia.fmt, .buf = x};
        rknn_output out = {.index = 0, .want_float = 1};
        if (rknn_inputs_set(ctx, 1, &in) || rknn_run(ctx, NULL) || rknn_outputs_get(ctx, 1, &out, NULL)) {
            fprintf(stderr, "run failed\n");
            return 1;
        }
        rknn_outputs_release(ctx, 1, &out);
        rknn_perf_run pr;
        rknn_query(ctx, RKNN_QUERY_PERF_RUN, &pr, sizeof pr);
        printf("PERF_RUN mask %s run %d run_duration_us %lld\n", m, r, (long long)pr.run_duration);
    }
    rknn_perf_detail pd;
    if (!rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &pd, sizeof pd) && pd.perf_data)
        fwrite(pd.perf_data, 1, pd.data_len, stdout);
    rknn_destroy(ctx);
    return 0;
}
