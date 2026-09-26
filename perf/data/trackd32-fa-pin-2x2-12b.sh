#!/bin/bash
# THE FLASH-ATTENTION OFFLOAD'S CURRENT WORTH AT pp2048, AND THE FIRST INTERACTION CELL WHOSE
# KNOB ADDS HOST WORK.
#
# WHAT IS SETTLED. `ROCKET_FLASH_ATTN` is on by default with the `n_kv` gate at 1024; the
# crossover was measured 2026-06-28 at ~2K (1.00x @512, 0.97x @1024, 1.02x @2048, 1.07x @4096) on
# an F16 model the comment does not name [ggml-rocket.cpp, the gate's own comment]. Three
# default-on host-cost cuts have landed since, and an absolute rate from a build two months old is
# not a cell identifier [a standing calibration prior]. The pinning-by-knob interaction model
# I = (1-a)(1-b)/(1-a-b+b*phi) has been fitted on six cells, EVERY one of which removes host work
# (phi > 0, a > 0) or touches neither (MM_ASYM). No cell has tested its sign structure with a knob
# of the opposite sign.
#
# THE ARM. `ROCKET_FLASH_ATTN=0` moves the above-gate FA ops from the NPU handler (batched QK/AV on
# the part, host mask+softmax on the worker pool, single-threaded gather/scatter) to the CPU
# backend's `flash_attn_ext_tiled` at `-t` threads. It is a SWAP, so its `a/phi` is the swap's
# exchange rate and NOT the attention path's -- trackd30 measures that directly. What this cell
# produces is (1) K_fa = t/s(FA on)/t/s(FA off) at pp2048 unpinned, the offload's current worth
# on this unit, and (2) a 2x2 whose knob has phi < 0 and a < 0.
#
# THE PREDICTION [registered 2026-09-03]. Below-gate CPU attention is 6.72% of pinned A76
# cycles = 10.5 core-s per prefill for the smallest quarter of the ops (n_kv 512). Scaling the
# tiled cost linearly in n_kv over Gemma-4's 40 local (window 1024) and 8 global layers puts the
# moved ops at ~6.5x that: ~68 core-s on the CPU against the ~25 core-s the handler's host side
# costs today, so phi_fa0 = -(68-25)/155.7 = -0.28 (band -0.20 to -0.35) [expected: linear
# scaling and equal per-element cost]. Wall: those 68 core-s at 4 threads are 17-20 s pinned
# against a handler interval of 11-19 s (trackd30's band), so the pinned swap is near break-even:
# K_pin 1.00-1.10. Unpinned the CPU path has 8 threads and the handler's five workers 8 cores:
# K_unpin 0.98-1.05, centred on the June 1.02. The model with a = 1 - 1/K_fa0 < 0 and phi < 0
# predicts I = K_fa(pin)/K_fa(unpin) > 1 -- the first cell with a predicted POSITIVE interaction:
# at a = -0.03, b = 0.055, phi = -0.28, I = 1.015; band 1.01-1.06. THE SIGN IS THE REGISTERED
# CLAIM. DECISION RULE: K_unpin < 1.00 at >= 3 se reopens the MIN_KV gate at pp2048 on this unit.
# RIVAL: I <= 1.00 refutes the model's sign structure for a knob that adds host work.
#
# WHAT A GREEN RESULT WOULD NOT SHOW. One unit, one shape; the gate's behaviour at 4K-16K, where
# the recorded win is, is untouched. And `q` from this cell is the swap's.
#
# BUDGET. The interaction's per-pass sd on this unit is 2.64%; three passes give se ~1.5%, so the
# sign claim resolves only if I lands above ~1.03. Read the se before calling the sign.
#
# CHECK THE <!--DATA--> ROW COUNT AFTER PASS 1: 4 arms, so 4 rows.
set -u
OUTD=${OUTD:-$PWD/trackd32-fa-pin-12b}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/gemma4/gemma-4-12b-it-F16.gguf"
export PINWRAP_BIN=${PINWRAP_BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export BIN=${BIN:-$MODELS/llama-bench-pinwrap.sh}
export TESTS=${TESTS:-"-p 2048 -n 0 -r 2"}
export PASSES=${PASSES:-3}
export OUT="$OUTD/gemma4-12b-f16-fa-pin2x2.md"
export ERRD="$OUTD/err"
export ARMS='fa1_unpin||
fa1_pin|PIN_MASK=0xf0|-t 4
fa0_unpin|ROCKET_FLASH_ATTN=0|
fa0_pin|ROCKET_FLASH_ATTN=0 PIN_MASK=0xf0|-t 4'
LOG="$OUTD/trackd32.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "NOT EXECUTABLE: $BIN" >&2; exit 2; }
[ -x "$PINWRAP_BIN" ] || { echo "MISSING: $PINWRAP_BIN" >&2; exit 2; }
[ -f "$SO" ]    || { echo "MISSING: $SO" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- gemma4-12b F16, pinning x ROCKET_FLASH_ATTN=0 -- the swap's worth and a negative-phi cell"
  echo "     TESTS='$TESTS'  PASSES=$PASSES"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     fa1_* = default (FA offload on, gate 1024); fa0_* = ROCKET_FLASH_ATTN=0 (CPU attention)"
  echo "     K_fa = fa1/fa0 in t/s, paired within a pass, unpinned and pinned"
  echo "     a = 1 - 1/(fa0_unpin/fa1_unpin) < 0; phi = 1 - busy_tot(fa0_pin)/busy_tot(fa1_pin) < 0"
  echo "     interaction I = (fa0_pin/fa1_pin) / (fa0_unpin/fa1_unpin) = K_fa(pin)/K_fa(unpin)"
  echo "     PREDICTION 2026-09-03f: K_unpin 0.98-1.05 (centre 1.02), K_pin 1.00-1.10,"
  echo "       phi -0.20 to -0.35, I > 1 (band 1.01-1.06) -- THE SIGN IS THE CLAIM."
  echo "       RULE: K_unpin < 1.00 at >= 3 se reopens the MIN_KV gate at pp2048."
  echo "       RIVAL: I <= 1.00."
  echo "     CONTROL: every arm streamed (0 resident lines)"
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 4 arms, RO on, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" gemma4-12b-f16-fa-pin2x2
  rc=$?
  echo "[$(date -Is)] rc=$rc TRACKD32-ALL-DONE"
  echo "--- residency control: every arm must report 0 resident ---"
  for f in "$ERRD"/*.err; do printf '%s: ' "$(basename "$f")"; grep -aoE '\[f16-resident\][^\n]*' "$f" | tail -1 || echo "(no resident line)"; done
} >> "$LOG" 2>&1
