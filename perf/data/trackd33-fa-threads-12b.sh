#!/bin/bash
# THREADING THE FA HANDLER'S FIVE HOST WALKS: THE A/B THE CAP WAS WRITTEN FOR.
#
# WHAT IS SETTLED. The `FLASH_ATTN_EXT` handler runs on the ggml dispatch thread while the
# scheduler waits, so its bracket IS wall and no exchange rate enters. On `gemma4-12b` F16 pinned
# and streamed at pp2048 the split reads gather 3.46 s, compute 16.99 s, scatter 1.98 s per
# 2048-token prefill of 97.8 s [ro-session/trackd30-fa-timing-12b.md]. The gather and scatter are
# the single-threaded strided->dense walks; their `q` is 1 by construction, which caps threading
# them over `k` workers at 5.6% x (1 - 1/k) of the wall with nothing imported.
#
# WHAT THIS MEASURES. `ROCKET_FA_THREADS=k` splits all five walks -- the Q convert, the K copy,
# the V transpose, the mask copy and the output scatter -- over the process-wide host pool, which
# is already pinned to the big cluster. `G_k` is gather+scatter seconds per 2048-token prefill at
# that setting, read from the `ROCKET FA total` line and NOT from the wall: the interval is what
# the band is written in and the wall is the secondary reading.
#
# WHY THREE ARMS AND NOT TWO. `k`=2 costs the same total board time as a fourth pass of a two-arm
# unit and answers a question the two-arm unit cannot: whether the gain saturates. If `G_2`/`G_1`
# is near 0.5 and `G_4`/`G_1` near 0.29 the walks scale with cores; if the two are close, memory
# bandwidth caps them and the second and fourth workers buy little. That is the discrimination
# the band's floor was argued from, so it is worth an arm rather than a pass.
#
# THE PREDICTION [registered 2026-09-07, before the code was written]. `r` = `G_4`/`G_1`
# with `G_1` = 5.44 s measured. Band `r` = 0.29-0.44 (`G_4` 1.6-2.4 s), worth 3.1-3.9% of the
# pinned wall, wall ratio `t_1`/`t_4` = 1.032-1.041. The axis is tiled by three thresholds, so
# every landing place is registered: below 0.29 the walks were compute-bound and the
# memory-traffic argument that set the floor is wrong; 0.44-0.63 says bandwidth caps the gain
# harder than the 2.4 s estimate; above 0.63 says the loops were never the bottleneck of their own
# interval and retires the 5.6% cap as a cap on THIS lever.
#
# THE CORRECTNESS CLAIM IS A CLAIM. Bit-identical at every `k` is an argument about disjoint index
# ranges, not a measurement. trackd33-fa-correctness.sh is the gate; it must pass at k=1,3,4
# before any number here is read.
#
# `-v` ON EVERY ARM: llama-bench installs a null ggml log callback unless verbose, and the probe's
# summary is a GGML_LOG_INFO line, so without -v it is silently lost. Every arm carries it, and
# every arm carries ROCKET_FA_TIMING, so neither is a difference between the arms.
#
# CHECK THE <!--DATA--> ROW COUNT: 3 arms x 3 passes = 9 rows. Then check that every arm's .err
# carries a `ROCKET FA total` line with the SAME op count -- an arm whose op count differs was
# not running the same graph, and its ratio is not a ratio of the same thing.
set -u
OUTD=${OUTD:-$PWD/trackd33-fathreads12b}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/gemma4/gemma-4-12b-it-F16.gguf"
export PINWRAP_BIN=${PINWRAP_BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export BIN=${BIN:-$MODELS/llama-bench-pinwrap.sh}
export TESTS=${TESTS:-"-p 2048 -n 0 -r 2"}
export PASSES=${PASSES:-3}
export OUT="$OUTD/gemma4-12b-f16-fathreads.md"
export ERRD="$OUTD/err"
export ARMS='fa_t1|ROCKET_FA_THREADS=1 ROCKET_FA_TIMING=1 PIN_MASK=0xf0|-t 4 -v
fa_t2|ROCKET_FA_THREADS=2 ROCKET_FA_TIMING=1 PIN_MASK=0xf0|-t 4 -v
fa_t4|ROCKET_FA_THREADS=4 ROCKET_FA_TIMING=1 PIN_MASK=0xf0|-t 4 -v'
LOG="$OUTD/trackd33.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "NOT EXECUTABLE: $BIN" >&2; exit 2; }
[ -x "$PINWRAP_BIN" ] || { echo "MISSING: $PINWRAP_BIN" >&2; exit 2; }
[ -f "$SO" ]    || { echo "MISSING: $SO" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- gemma4-12b F16, PINNED (PIN_MASK=0xf0 -t 4), streamed: threading the FA handler's host walks"
  echo "     TESTS='$TESTS'  PASSES=$PASSES  every arm -v and ROCKET_FA_TIMING=1, so neither is a difference"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     so=$(md5sum "$SO" | cut -d' ' -f1)"
  echo "     G_k = (gather+scatter) ms from the ROCKET FA total line, divided by the prefill count"
  echo "     PREDICTION 2026-09-07: r = G_4/G_1 = 0.29-0.44, wall ratio t_1/t_4 = 1.032-1.041."
  echo "       Thresholds: r < 0.29 refutes the memory-traffic floor; 0.44-0.63 says bandwidth caps"
  echo "       it harder; r > 0.63 retires the 5.6% cap as a cap on THIS lever."
  echo "     CONTROL: every arm streamed (0 resident lines) and every arm the same FA op count"
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 3 arms, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" gemma4-12b-f16-fathreads
  rc=$?
  echo "[$(date -Is)] rc=$rc TRACKD33-ALL-DONE"
  echo "--- FA timing lines, one per arm per pass ---"
  for f in "$ERRD"/*.err; do
    printf '%s: ' "$(basename "$f" .err)"
    grep -a "ROCKET FA total" "$f" | tail -1 || echo "(no FA total line)"
  done
  echo "--- residency control: every arm must report 0 resident ---"
  for f in "$ERRD"/*.err; do
    printf '%s: ' "$(basename "$f" .err)"
    grep -aoE '\[f16-resident\][^\n]*' "$f" | tail -1 || echo "(no resident line)"
  done
} >> "$LOG" 2>&1
