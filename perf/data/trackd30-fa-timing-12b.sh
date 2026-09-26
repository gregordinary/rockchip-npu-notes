#!/bin/bash
# THE FLASH-ATTENTION HANDLER'S CRITICAL-PATH INTERVAL, READ DIRECTLY, WITHOUT AN EXCHANGE RATE.
#
# WHAT IS SETTLED. On `gemma4-12b` F16 pinned and streamed, the attention path is 22.93% of
# non-idle A76 cycles, the same size as the host weight pack, and its 5.85% cap is the one row in
# the cap table that IMPORTS another knob's exchange rate (the pack's `a/phi` = 0.255, `q` = 7.73)
# [ro-session/trackd27-a76-ipc-12b.md; tuning-matrix.md]. `q` is knob-dependent by construction,
# 4.46-7.73 over four cells, so that cap is unbounded in either direction until the term's own
# critical-path cost is measured.
#
# WHY NOT `ROCKET_FLASH_ATTN=0`. Attention has to be computed somewhere, so every FA knob is a SWAP
# between two implementations and never a deletion: its `a` is the difference of two paths and its
# `q` is the swap's, not the attention path's. A `q` derived from it cannot multiply the 22.93%.
#
# WHAT IS MEASURED INSTEAD. The handler runs on the single backend dispatch thread while the
# scheduler waits (ggml runs backend splits sequentially and the rocket backend is synchronous), so
# the handler's own bracket IS its critical-path interval -- the one case where a bucket's thread
# interval is a share of the wall. `ROCKET_FA_TIMING=1` prints gather / compute / scatter ms over
# every offloaded FA op at exit: gather and scatter are the single-threaded strided->dense loops
# on the dispatch thread; compute is `rocket_flash_attn_fp16_ctx` (worker fan-out: batched QK on
# the NPU, host mask+softmax on the workers, batched AV on the NPU). It does NOT cover the
# below-gate ops (`n_kv` < 1024, the first 512-token micro-batch of every layer) that the CPU
# backend runs as `ggml_compute_forward_flash_attn_ext_tiled`, 6.72% of cycles.
#
#   share = (gather+compute+scatter) / ((r+1) * 2048 / t_s)   the timed process runs r+1 prefills
#
# THE PREDICTION [registered 2026-09-03]. The handler symbol's OWN leaf cycles are 1.92% of
# non-idle A76 cycles, and the pinned arm's host CPU is 155.7 core-s per prefill (busy_tot 62300
# jiffies over 4 prefills, trackd-pin12b pin76t4), so gather+scatter = 3.0 core-s per prefill, and
# single-threaded core-seconds are wall 1:1: 2.4-3.4 s per prefill = 2.5-3.5% of the ~97 s wall.
# compute: `host_softmax_rows` 3.35% + `fa_mask_scores` 2.86% + `expf` (the CPU tiled path has no
# scalar expf, so most of the 7.72% is the host softmax's) ~= 14% of cycles = ~22 core-s per prefill
# on the worker pool, plus device QK/AV time: 8-16 s (8-17%). Total 11-19 s = 11-20% of wall.
# Probe overhead: the probe arm's t/s within 1% of the plain arm's (two clock reads per op).
# DECISION RULE: total < 12% closes a vectorised-exp build below 4% of prefill wall (exp is at
# most half the handler's host cycles and the host part is at most all of compute); >= 12% keeps
# it open and the next step is a caller split of `expf`. RIVAL: gather+scatter > 5% says the
# handler's leaf cycles under-count it -- the kernel's two unnamed 6.77% and 5.91% entries would
# then be page-fault / copy work inside the gather, attributed to the kernel by leaf IP.
#
# WHAT A GREEN RESULT WOULD NOT SHOW. One unit, one shape, one worker count. compute mixes device
# time with the host softmax and this probe cannot split them. And the share is of the PINNED
# wall; the campaign numbers are unpinned.
#
# `-v` ON BOTH ARMS: llama-bench installs a null log callback unless verbose, and the probe's
# summary is a GGML_LOG_INFO line, so without -v it is silently lost. Both arms carry it so the
# flag is not itself a difference between them.
#
# CHECK THE <!--DATA--> ROW COUNT: 2 arms, 1 pass, so 2 rows. Then grep the probe arm's .err for
# "ROCKET FA total".
set -u
OUTD=${OUTD:-$PWD/trackd30-fatiming12b}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/gemma4/gemma-4-12b-it-F16.gguf"
export PINWRAP_BIN=${PINWRAP_BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export BIN=${BIN:-$MODELS/llama-bench-pinwrap.sh}
export TESTS=${TESTS:-"-p 2048 -n 0 -r 2"}
export PASSES=${PASSES:-1}
export OUT="$OUTD/gemma4-12b-f16-fatiming.md"
export ERRD="$OUTD/err"
export ARMS='fa_plain|PIN_MASK=0xf0|-t 4 -v
fa_probe|ROCKET_FA_TIMING=1 PIN_MASK=0xf0|-t 4 -v'
LOG="$OUTD/trackd30.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "NOT EXECUTABLE: $BIN" >&2; exit 2; }
[ -x "$PINWRAP_BIN" ] || { echo "MISSING: $PINWRAP_BIN" >&2; exit 2; }
[ -f "$SO" ]    || { echo "MISSING: $SO" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- gemma4-12b F16, PINNED (PIN_MASK=0xf0 -t 4), streamed: the FA handler's critical-path interval"
  echo "     TESTS='$TESTS'  PASSES=$PASSES  both arms -v so GGML_LOG_INFO reaches stderr"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     fa_plain = the published pin76t4 arm; fa_probe = the same + ROCKET_FA_TIMING=1"
  echo "     share = (gather+compute+scatter) / ((r+1)*2048/t_s)"
  echo "     PREDICTION 2026-09-03d: gather+scatter 2.5-3.5% of wall, compute 8-17%, total 11-20%;"
  echo "       probe overhead < 1% of t/s. RULE: total < 12% closes a vectorised-exp build below 4%."
  echo "       RIVAL: gather+scatter > 5% -> the kernel's unnamed 6.77%/5.91% entries are the gather's."
  echo "     CONTROL: both arms streamed (0 resident lines)"
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 2 arms, RO on, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" gemma4-12b-f16-fatiming
  rc=$?
  echo "[$(date -Is)] rc=$rc TRACKD30-ALL-DONE"
  echo "--- FA timing lines ---"
  grep -a "ROCKET FA total" "$ERRD"/*.err || echo "(no FA total line: -v did not unlock the log, or the probe never armed)"
  echo "--- residency control: every arm must report 0 resident ---"
  for f in "$ERRD"/*.err; do printf '%s: ' "$(basename "$f")"; grep -aoE '\[f16-resident\][^\n]*' "$f" | tail -1 || echo "(no resident line)"; done
} >> "$LOG" 2>&1
