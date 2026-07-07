#!/usr/bin/env python3
# Phase 2 (prove): patch whisper.cpp so each encoder layer's whole block runs on the NPU
# via one ggml_map_custom1 -> rocket_encoder_block_fp16. Two gates:
#   build: -DWHISPER_ROCKET=ON  (default OFF -> stock build unaffected)
#   run  : WHISPER_ROCKET_ENC=1 (default off -> stock encoder)
# Idempotent; backs up originals once.
import sys, os
ROOT = sys.argv[1] if len(sys.argv) > 1 else "whisper.cpp"
SRC = f"{ROOT}/src/whisper.cpp"
CM  = f"{ROOT}/src/CMakeLists.txt"

s = open(SRC).read()
if "WHISPER_ROCKET_ENABLED" in s:
    print("whisper.cpp already patched")
else:
    if not os.path.exists(SRC + ".bak"): open(SRC + ".bak", "w").write(s)
    # 1) rocket callback block, inserted before the encoder graph builder
    anchor1 = "static struct ggml_cgraph * whisper_build_graph_encoder("
    assert s.count(anchor1) == 1, f"anchor1 count={s.count(anchor1)}"
    block = r'''#ifdef WHISPER_ROCKET_ENABLED
extern "C" {
#include "rocket_npu.h"
#include "rocket_encoder.h"
}
namespace {
typedef _Float16 wre_f16;
struct wre_layer {
    bool ready;
    const wre_f16 *wq,*wk,*wv,*wo,*wf1,*wf2;                 // F16 weights, used in place
    wre_f16 *ln1g,*ln1b,*bq,*bv,*bo,*ln2g,*ln2b,*bf1,*bf2;   // F32->fp16, converted once
    int nh, dff; float eps;
};
static int       g_wre_fd = -2;
static wre_layer g_wre[256];
static wre_f16 * wre_cvt(const struct ggml_tensor * t) {     // F32 tensor -> fresh fp16 buffer
    int64_t n = ggml_nelements(t); wre_f16 * o = (wre_f16 *) malloc(n * sizeof(wre_f16));
    const float * sp = (const float *) t->data; for (int64_t i = 0; i < n; i++) o[i] = (wre_f16) sp[i];
    return o;
}
static void wre_cb(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * ud) {
    (void) nth; if (ith != 0) return;                        // rocket is internally serial; one task
    wre_layer * L = (wre_layer *) ud;
    int d = (int) a->ne[0], T = (int) a->ne[1]; size_t Td = (size_t) T * d;   // a is [n_state, n_ctx] = [d,T] -> mem [T][d]
    wre_f16 * in  = (wre_f16 *) malloc(Td * sizeof(wre_f16));
    wre_f16 * out = (wre_f16 *) malloc(Td * sizeof(wre_f16));
    const float * af = (const float *) a->data; for (size_t i = 0; i < Td; i++) in[i] = (wre_f16) af[i];
    rocket_encoder_block_fp16(g_wre_fd, T, d, L->nh, L->dff, in,
        L->ln1g, L->ln1b, L->wq, L->bq, L->wk, NULL, L->wv, L->bv, L->wo, L->bo,
        L->ln2g, L->ln2b, L->wf1, L->bf1, L->wf2, L->bf2, L->eps, out);
    float * df = (float *) dst->data; for (size_t i = 0; i < Td; i++) df[i] = (float) out[i];
    free(in); free(out);
}
} // namespace
#endif

'''
    s = s.replace(anchor1, block + anchor1, 1)
    # 2) loop-site: divert each encoder layer to the fused NPU block
    anchor2 = "        const auto & layer = model.layers_encoder[il];"
    assert s.count(anchor2) == 1, f"anchor2 count={s.count(anchor2)}"
    divert = anchor2 + r'''
#ifdef WHISPER_ROCKET_ENABLED
        {
            static int wre_en = -1; if (wre_en < 0) wre_en = getenv("WHISPER_ROCKET_ENC") ? 1 : 0;
            if (wre_en) {
                wre_layer * WL = &g_wre[il];
                if (!WL->ready) {
                    if (g_wre_fd == -2) { g_wre_fd = rocket_open(); fprintf(stderr, "[whisper-rocket] encoder fd=%d\n", g_wre_fd); }
                    WL->nh = n_head; WL->dff = (int) layer.mlp_0_w->ne[1]; WL->eps = hparams.eps;
                    WL->wq = (const wre_f16 *) layer.attn_q_w->data; WL->wk = (const wre_f16 *) layer.attn_k_w->data;
                    WL->wv = (const wre_f16 *) layer.attn_v_w->data; WL->wo = (const wre_f16 *) layer.attn_ln_1_w->data;
                    WL->wf1 = (const wre_f16 *) layer.mlp_0_w->data; WL->wf2 = (const wre_f16 *) layer.mlp_1_w->data;
                    WL->ln1g = wre_cvt(layer.attn_ln_0_w); WL->ln1b = wre_cvt(layer.attn_ln_0_b);
                    WL->bq = wre_cvt(layer.attn_q_b); WL->bv = wre_cvt(layer.attn_v_b); WL->bo = wre_cvt(layer.attn_ln_1_b);
                    WL->ln2g = wre_cvt(layer.mlp_ln_w); WL->ln2b = wre_cvt(layer.mlp_ln_b);
                    WL->bf1 = wre_cvt(layer.mlp_0_b); WL->bf2 = wre_cvt(layer.mlp_1_b);
                    WL->ready = true;
                }
                inpL = ggml_map_custom1(ctx0, inpL, wre_cb, 1, WL);
                continue;
            }
        }
#endif'''
    s = s.replace(anchor2, divert, 1)
    open(SRC, "w").write(s)
    print("patched whisper.cpp")

# 3) CMake: optional link of librocketnpu
c = open(CM).read()
if "WHISPER_ROCKET" in c:
    print("CMakeLists already patched")
else:
    if not os.path.exists(CM + ".bak"): open(CM + ".bak", "w").write(c)
    canchor = "target_link_libraries(whisper PUBLIC ggml Threads::Threads)"
    assert c.count(canchor) == 1, f"cmake anchor count={c.count(canchor)}"
    cblock = canchor + r'''

# --- experimental: fused on-NPU Whisper encoder via librocketnpu (env WHISPER_ROCKET_ENC=1) ---
option(WHISPER_ROCKET "link librocketnpu for the fused on-NPU Whisper encoder" OFF)
if (WHISPER_ROCKET)
    if (NOT DEFINED ROCKETNPU_DIR)
        set(ROCKETNPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../rocket-userspace")
    endif()
    target_include_directories(whisper PRIVATE ${ROCKETNPU_DIR}/include /usr/include/libdrm)
    target_link_libraries(whisper PRIVATE ${ROCKETNPU_DIR}/build_nv/librocketnpu.a drm)
    target_compile_definitions(whisper PRIVATE WHISPER_ROCKET_ENABLED)
endif()'''
    c = c.replace(canchor, cblock, 1)
    open(CM, "w").write(c)
    print("patched src/CMakeLists.txt")
print("DONE")
