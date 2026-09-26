#!/bin/bash
# THE RESIDENCY INGEST'S CPU COST, AND THE UNCONTAMINATED HOST SHARE OF A RESIDENCY KNOB.
#
# WHAT IS BEING TESTED. Four of five pinning-by-knob 2x2 cells on this board are reproduced with
# no free parameter by writing the wall as a host term plus a rest that pinning cannot touch:
#
#     I = (1 - a)(1 - b) / (1 - a - b + b*phi)
#
# with a = 1 - 1/K (K the knob's unpinned paired ratio), b = 1 - 1/P (P the base pin gain), and
# phi the fraction of HOST core-seconds the knob removes, read from `busy_tot` in the two pinned
# arms. Residuals: +0.1 se (`qwen35-9b` x ub), +1.5 and -1.0 se (`gemma4-12b` F16 x `MM_ASYM`),
# -1.1 se (`qwen35-9b` x quant residency), and **-3.5 se** on `gemma4-12b` F16 x f16 residency.
#
# WHY THE MISS IS PROBABLY THE INSTRUMENT [hypothesis]. The residual is ordered by how much
# NON-TIMED wall each knob's arm adds: 2.5 vs 3.6 s and 5.4 vs 5.5 s at +0.1 and +1.5 se, 2.4 vs
# 37.3 s at -1.1, and 5.4 vs 158.8 s at -3.5. `bench-llm.sh` takes `wall_s` and both `busy_tot`
# snapshots around the WHOLE `llama-bench` invocation, and a residency arm pays its one-time
# ingest inside that bracket -- the shell warm-up that runs before the readout opens is a separate
# process, so the timed process places its weights again. So `phi` is biased LOW for exactly the
# two cells that miss, in exactly the direction of their residuals.
#
# THE ARM. Not a 2x2. `busy_tot = intercept + (r+1)*slope`, because llama-bench runs one internal
# warm-up plus `r` reps. Regressing `busy_tot` on the rep count separates them:
#
#   slope      = per-rep host work        -> phi = 1 - slope_res/slope_stream, uncontaminated
#   intercept  = model load + the ingest  -> the ingest itself, as core-seconds
#
# Six arms, PINNED only (`PIN_MASK=0xf0 -t 4`; phi is defined at fixed pinning), streamed against
# `ROCKET_F16_RESIDENT=auto`, at `-r 1`, `-r 2`, `-r 4`.
#
# WHY NOT A SHORT-PROMPT PROBE. Reading the intercept from a `-p 128` arm would be four times
# cheaper and would measure a different cell: a shorter prompt means smaller compute buffers, so
# MORE weights place before the reserve floor latches, so the ingest under test changes. This
# holds `-p 2048` and varies only the rep count.
#
# THE PREDICTION, REGISTERED BEFORE THE RUN, 2026-09-02: slope-based
# phi = **0.28** (band 0.16-0.40) against the contaminated 0.087; intercept difference =
# **120 core-seconds** (band 60-190). RIVAL: the MoE route's expert pack is bytes-bound at
# 505 MB/s [perf/benchmarks.md], which over this knob's 18078 MB is 36 core-seconds
# single-threaded -- below the band, and a reading there says the f16 ingest is a different,
# cheaper term and the model's miss has another cause.
#
# NOT A PERF LEVER, SO NO CAP APPLIES. The output is the trustworthiness of `busy_tot`, a column
# every campaign in this workspace writes.
#
# THE CONTROL THAT MUST SUCCEED. Placement has to be CONSTANT across the three rep counts, or the
# regression is fitting three different configurations. Every res arm's `[f16-resident]` teardown
# line is grepped below and they must agree; the stream arms must report zero placed. An arm that
# disagrees is not this contrast.
#
# POSITION IS UNBALANCED AND THAT IS ACCEPTED HERE. Six arms rotated by one over three passes give
# each arm three of six positions. The position effect measured on this board is +-0.3% of t/s
# [ro-session/trackd23*], against a predicted phi difference of 0.087 to 0.28 -- a factor of three.
# Balancing it would cost six passes and buy nothing at this effect size.
#
# CHECK THE <!--DATA--> ROW COUNT AFTER PASS 1: 6 arms, so 6 rows.
set -u
OUTD=${OUTD:-$PWD/trackd25-busyint12b}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/gemma4/gemma-4-12b-it-F16.gguf"
export PINWRAP_BIN=${PINWRAP_BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export BIN=${BIN:-$MODELS/llama-bench-pinwrap.sh}
# -r lives in each arm's args, so TESTS must NOT carry one.
export TESTS=${TESTS:-"-p 2048 -n 0"}
export PASSES=${PASSES:-3}
export OUT="$OUTD/gemma4-12b-busy-intercept.md"
export ERRD="$OUTD/err"
export ARMS='stream_r1|PIN_MASK=0xf0|-t 4 -r 1
stream_r2|PIN_MASK=0xf0|-t 4 -r 2
stream_r4|PIN_MASK=0xf0|-t 4 -r 4
res_r1|ROCKET_F16_RESIDENT=auto PIN_MASK=0xf0|-t 4 -r 1
res_r2|ROCKET_F16_RESIDENT=auto PIN_MASK=0xf0|-t 4 -r 2
res_r4|ROCKET_F16_RESIDENT=auto PIN_MASK=0xf0|-t 4 -r 4'
LOG="$OUTD/trackd25.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "NOT EXECUTABLE: $BIN" >&2; exit 2; }
[ -x "$PINWRAP_BIN" ] || { echo "MISSING: $PINWRAP_BIN" >&2; exit 2; }
[ -f "$SO" ]    || { echo "MISSING: $SO" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- gemma4-12b F16, busy_tot against rep count, PINNED arms only"
  echo "     TESTS='$TESTS'  PASSES=$PASSES  (-r lives in each arm's args)"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     stream_rN = the shipping default; res_rN = ROCKET_F16_RESIDENT=auto"
  echo "     all arms PIN_MASK=0xf0 -t 4; phi is defined at fixed pinning"
  echo "     busy_tot = intercept + (r+1)*slope; llama-bench runs one internal warm-up plus r reps"
  echo "     phi = 1 - slope_res/slope_stream ; ingest = intercept_res - intercept_stream"
  echo "     prediction 2026-09-02e: phi=0.28 (0.16-0.40), ingest=120 core-s (60-190)"
  echo "     CONTROL: every res arm must place the SAME weight count; every stream arm zero"
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 6 arms, RO on, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" gemma4-12b-busy-intercept
  rc=$?
  echo "[$(date -Is)] rc=$rc TRACKD25-ALL-DONE"
  echo "--- placement control: every res arm must agree, every stream arm must be zero ---"
  for f in "$ERRD"/*.err; do
    printf '%s: ' "$(basename "$f")"
    grep -oE '\[f16-resident\][^\n]*' "$f" | tail -1 || echo "(no f16-resident line)"
  done
} >> "$LOG" 2>&1
