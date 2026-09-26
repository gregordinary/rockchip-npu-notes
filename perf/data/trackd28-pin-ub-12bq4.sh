#!/bin/bash
# FILL THE `phi` GAP, AND ASK WHETHER THE HOST SHARE IS A PROPERTY OF THE MODEL OR OF THE GGUF.
#
# WHAT IS SETTLED. `H/t = a/phi` returns the host term's share of prefill wall: a knob's paired
# ratio divided by the fraction of host core-seconds it removes. Three cells are measured --
# `qwen35-9b` Q4_K_M reads 0.669 (quant residency) and 0.591 (`-b 2048 -ub 2048`), `gemma4-12b`
# F16 reads 0.255 [tuning-matrix.md]. The `phi` axis holds 0.025-0.029, 0.087, 0.227, 0.496 and
# 0.598, and NOTHING between 0.09 and 0.50.
#
# WHY THIS UNIT. `gemma4-12b` names two GGUFs here, an F16 one at ~20 t/s at pp2048 and a Q4_K_M
# one at 12.87, and the whole flag table is the quant class. The F16 unit's host share is measured
# and the Q4_K_M unit's is not, so the pair holds the MODEL fixed and varies the GGUF -- which is
# the only contrast that can say which of the two the host share belongs to.
#
# THE BUDGET COMES FROM `phi`, NOT FROM THE INTERACTION. `-b 2048 -ub 2048` places no resident
# weights, so its `busy_tot` carries no one-time ingest and its `phi` needs no rep-count
# regression: it is a plain ratio of the two PINNED arms, measured at se 0.0011 over six passes on
# the 9B. The interaction is a DIFFERENT claim at a 2.64% per-pass sd, and under the leading
# hypothesis it sits 1.4 pp from the null, which three passes cannot read. Record it, report its
# se, and do not read it.
#
#     phi         = 1 - busy_tot(ub_pin) / busy_tot(stock_pin)
#     a           = 1 - 1/K, K the UNPINNED paired ratio
#     H/t         = a / phi          (a share of the UNPINNED wall, because `a` is unpinned)
#     R           = t * (1 - a/phi)  the device term, t = 2048 / stock_unpin t/s
#     interaction = (ub_pin/stock_pin) / (ub_unpin/stock_unpin), paired within a pass
#
# THE PREDICTION, IN THE UNIT THAT TRANSFERS [registered 2026-09-03]. `R` is a property of
# the MODEL: both GGUFs run the same fp16 GEMMs on the part and differ only by a host-side
# dequant. F16 stock 19.98 t/s at `H/t` = 0.255 gives `R` = 76.4 s per 2048-token prefill, so this
# unit's 159.1 s wall is 76 s of device and ~83 s of host. That forces `H/t` = 0.520 and
# `phi` = 0.393, band 0.34-0.46. RIVAL A, the host share tracks the MODEL at 0.255: `phi` = 0.802.
# RIVAL B, it tracks the QUANT CLASS at the 9B's 0.591: `phi` = 0.346. All three are tens of se
# apart, so THIS CELL CANNOT COME BACK UNRESOLVED.
#
# NOT A PERF LEVER, SO NO CAP APPLIES. The output is a denominator.
#
# WHAT A GREEN RESULT WOULD NOT SHOW. `R` transferring between two GGUFs of one model says nothing
# about it transferring across models. One shape, one build. And this does not re-test the
# interaction model -- five cells already carry that.
#
# A COMMAND WRAPPER CANNOT RIDE IN THE ENV FIELD -- bench-llm.sh builds `env $envs
# ROCKET_LOG_STDERR=1 $BIN ...`, so a `taskset` there eats the assignment after it and the arm
# dies rc=127 in 46 seconds while still writing a full plausible <!--RO--> line. The mask travels
# as PIN_MASK and BIN points at the wrapper for EVERY arm, pinned or not.
#
# READ THE ENGAGEMENT LINE PER ARM. NO arm here should place anything: neither stock nor
# `-b 2048 -ub 2048` is a residency knob, and an arm that reports resident weights is not this
# contrast. That assertion is what makes the `phi` clean.
#
# THREE PASSES ON A FOUR-ARM ROTATION IS DELIBERATE HERE. A four-arm rotation wants a multiple of
# four passes, and three gives each arm three of the four positions. That bias is budgeted against
# the deliverable rather than against the interaction: the position effect on this board is +-0.3%
# of t/s [ro-session/trackd23*], and `phi` = 1 - busy(ub_pin)/busy(stock_pin) scales it by
# (1-phi)/phi, so ~0.2% relative on `phi` against candidate separations of 13% and 100%. The
# INTERACTION does carry the unbalanced rotation, which is a second reason not to read it here.
#
# CHECK THE <!--DATA--> ROW COUNT AFTER PASS 1: 4 arms, so 4 rows.
set -u
OUTD=${OUTD:-$PWD/trackd28-pinub12bq4}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
MODEL="$MODELS/gemma4/gemma-4-12b-it-Q4_K_M.gguf"
export PINWRAP_BIN=${PINWRAP_BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export BIN=${BIN:-$MODELS/llama-bench-pinwrap.sh}
export TESTS=${TESTS:-"-p 2048 -n 0 -r 2"}
export PASSES=${PASSES:-3}
export OUT="$OUTD/gemma4-12b-q4-ub-pin2x2.md"
export ERRD="$OUTD/err"
export ARMS='stock_unpin||
stock_pin|PIN_MASK=0xf0|-t 4
ub_unpin||-b 2048 -ub 2048
ub_pin|PIN_MASK=0xf0|-b 2048 -ub 2048 -t 4'
LOG="$OUTD/driver.log"

mkdir -p "$ERRD"
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "NOT EXECUTABLE: $BIN" >&2; exit 2; }
[ -x "$PINWRAP_BIN" ] || { echo "MISSING: $PINWRAP_BIN" >&2; exit 2; }
[ -f "$SO" ]    || { echo "MISSING: $SO" >&2; exit 2; }
: > "$OUT"
{
  echo "<!-- gemma4-12b Q4_K_M, pinning x -b 2048 -ub 2048 -- the phi-gap cell"
  echo "     TESTS='$TESTS'  PASSES=$PASSES"
  echo "     gguf=$MODEL ($(stat -c %s "$MODEL") bytes)"
  echo "     stock_unpin | stock_pin = PIN_MASK=0xf0 -t 4"
  echo "     ub_unpin    = -b 2048 -ub 2048   (the published 1.257x arm, Q4_K_M row)"
  echo "     ub_pin      = the same + PIN_MASK=0xf0 -t 4"
  echo "     phi         = 1 - busy_tot(ub_pin)/busy_tot(stock_pin), the two PINNED arms"
  echo "     a           = 1 - 1/K, K the UNPINNED paired ratio; H/t = a/phi; R = t*(1 - a/phi)"
  echo "     interaction = (ub_pin/stock_pin) / (ub_unpin/stock_unpin), paired within a pass"
  echo "     PREDICTION 2026-09-03b: R = 76 s (68-85), H/t = 0.520, phi = 0.393 (0.34-0.46)"
  echo "       RIVAL A host share tracks the MODEL (0.255) -> phi = 0.802"
  echo "       RIVAL B host share tracks the QUANT CLASS (0.591) -> phi = 0.346"
  echo "     CONTROL: no arm may place resident weights; one that does is not this contrast"
  echo "-->"
} >> "$OUT"
{
  echo "[$(date -Is)] start: PASSES=$PASSES, 4 arms, RO on, so=$(md5sum "$SO" | cut -d' ' -f1)"
  bash "$MODELS/bench-llm.sh" "$MODEL" gemma4-12b-q4-ub-pin2x2
  rc=$?
  echo "[$(date -Is)] rc=$rc ALL-DONE"
} >> "$LOG" 2>&1
