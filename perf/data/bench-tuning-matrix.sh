#!/bin/bash
# bench-tuning-matrix.sh [MODEL_KEY ...]      -- one model family per unit, run them one at a time
# bench-tuning-matrix.sh --list               -- the units, their arms, and their estimated cost
#
# The FLAG axis of the tuning matrix: per model, paired stock-default -> recommended, so that
# TUNING.md's recipes stop being projections. Drives bench-llm.sh's ARMS mode; this file holds
# only WHICH arms each model class gets and WHY, which is the part that has to be decided per
# model rather than per run.
#
# ONE MODEL PER INVOCATION, DELIBERATELY. The full matrix is hours of board time, so it is
# multi-session by construction: land a model's rows, commit them, stop. A driver that swept
# everything in one process would be a sweep whose tail cannot be finished, and a truncated
# sweep reads as "covered everything".
#
# WHAT "STOCK" MEANS. The absence of settings -- llama-bench's own defaults, which are what a
# plain llama-cli user gets with GGML_BACKEND_PATH set and nothing else: -b 2048 -ub 512, every
# ROCKET_* knob at its default (which since 2026-08-27 includes MoE placement on AUTO). Stock is
# spelled as an EMPTY env and args field rather than as flags that happen to match the defaults,
# so that a default moving underneath this file changes the measurement instead of hiding in it.
#
# WHAT THE ARMS ARE, per model class:
#   quant  stock -> ub2048 -> qresident            the -ub lever, then fp16 residency on top
#   f16    f16-stock -> f16-res                     ROCKET_F16_RESIDENT=auto; no -ub lever,
#                                                   because that tax is the quant re-dequant
#   moe    stock -> ub2048 -> moe1 -> moe0          moe0 is the MECHANISM control: it separates
#                                                   what the expert route buys from what the
#                                                   -ub lever buys, which stock-vs-ub2048 alone
#                                                   cannot
#   dqc    stock -> ub2048 -> dqc512 -> dqc2048     the DENSE mechanism control (the moe0
#                                                   analogue): ROCKET_DEQUANT_CACHE_MB holds the
#                                                   dequantized fp16 host-side, so the dqc pair
#                                                   prices the -ub lever with the dequant term
#                                                   removed; its 4th UNITS field is the budget MB
#   qres   stock -> qres512                         the UNSTACKED recipe: residency at the
#                                                   DEFAULT -ub, which is what a model whose
#                                                   non-dequant `-ub` residue is a LOSS should be
#                                                   run with. Two arms, not three: the point is
#                                                   that `-ub 2048` is ABSENT, so an arm carrying
#                                                   it would measure the stacked recipe again
#
# THE CPU ARM IS OFF BY DEFAULT (CPUARM=1 adds it), and that is a cost decision worth stating
# because it is most of the saving. "What does tuning buy" is a ratio of two NPU arms; the CPU
# arm answers a different question, is already published per model in perf/data/*.md at this
# clock and governor, and is the SLOWEST arm on exactly the models that cost the most -- one CPU
# pp2048 arm on Qwen3.6-27B is 77 minutes against 18 for the NPU arm beside it. Turn it on for a
# model whose CPU comparison is the point, not as a matter of course.
#
# THE GOVERNOR. bench-llm.sh refuses to run while a cpufreq policy is unpinned, and every row the
# matrix holds was taken unpinned. So reproducing a recorded row needs ALLOW_UNPINNED=1, and a
# pinned run is a new row rather than a re-measurement of an old one.
#
# COST. Default MODE=headline is the pp2048 row only -- the "how much does tuning buy" table is
# a pp2048 table, and paying for the curve on every arm is what turns a model into an hour.
# MODE=curve adds pp512/1024. Set MODE=curve only for a model whose crossover you are reading.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
GEN=${GEN:-$HERE/bench-llm.sh}
MODELS=${MODELS:?set MODELS to the directory holding the GGUFs}
export SO=${SO:-$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so}
export BIN=${BIN:-$HOME/npu/llama.cpp/build/bin/llama-bench}
OUTD=${OUTD:-$MODELS/tuning-matrix}
MODE=${MODE:-headline}
case "$MODE" in
  headline) export TESTS="-p 2048 -n 0 -r 3" ;;
  curve)    export TESTS="-p 512,1024,2048 -n 0 -r 2" ;;
  *) echo "MODE must be headline or curve" >&2; exit 2 ;;
esac

CPUARM=${CPUARM:-0}
arms_quant() {  # $1 = 1 if the fp16 resident copy fits beside the mmapped GGUF, else 0
  [ "$CPUARM" = 1 ] && printf 'cpu||\n'
  printf 'stock||\n'
  printf 'ub2048||-b 2048 -ub 2048\n'
  [ "$1" = 1 ] && printf 'qresident|ROCKET_QUANT_RESIDENT=auto|-b 2048 -ub 2048\n'
  return 0
}
arms_f16() {
  printf 'f16-stock||\n'
  printf 'f16-res|ROCKET_F16_RESIDENT=auto|\n'
}
arms_moe() {
  [ "$CPUARM" = 1 ] && printf 'cpu||\n'
  printf 'stock||\n'
  printf 'ub2048||-b 2048 -ub 2048\n'
  printf 'moe1|ROCKET_MOE=1|-b 2048 -ub 2048\n'
  printf 'moe0|ROCKET_MOE=0|-b 2048 -ub 2048\n'
}
# The unstacked recipe. Sub-4B quant models carry a 14-21% NON-dequant loss at `-ub 2048` that
# the dequant win masks, so for that class residency REPLACES the flag rather than stacking on it
# (llama32-3b 1.472x unstacked against 1.212x stacked, and three more like it). This arm set is
# what measures that, and it is a class recipe rather than a per-model curiosity.
arms_qres() {
  [ "$CPUARM" = 1 ] && printf 'cpu||\n'
  printf 'stock||\n'
  printf 'qres512|ROCKET_QUANT_RESIDENT=auto|\n'
  return 0
}
arms_dqc() {  # $1 = ROCKET_DEQUANT_CACHE_MB budget for this model (sized over its fp16 image)
  # The dense MECHANISM CONTROL for the -ub lever, mirroring the moe units' moe0 arm: the
  # dqc arms hold the dequantized fp16 weights in HOST RAM (dequant once per weight), so the
  # per-micro-batch dequant is gone while the per-call pack/upload and the placement stay the
  # shipped streaming path. dqc512-vs-stock is what the dequant COSTS at the default -ub;
  # dqc2048/dqc512 (paired within a pass) is what the -ub flag buys BEYOND dequant
  # amortization; (ub2048/stock)/(dqc2048/dqc512) is the dequant share of the headline lever.
  printf 'stock||\n'
  printf 'ub2048||-b 2048 -ub 2048\n'
  printf 'dqc512|ROCKET_DEQUANT_CACHE_MB=%s|\n' "$1"
  printf 'dqc2048|ROCKET_DEQUANT_CACHE_MB=%s|-b 2048 -ub 2048\n' "$1"
}

# key | gguf (relative to $MODELS) | class | fp16-resident-fits | note
#
# The fp16-fits column is the RAM math from TUNING.md against this 31 GiB board, and it is the
# reason a unit is cheap or is a question: a quant model's resident arm needs its fp16 size to
# sit beside the still-mapped GGUF. Gemma-4-12B is the boundary the tuning gaps name explicitly
# ("untested whether a 12B+ fp16 resident even fits"), so its arm RUNS and the refusal, if that
# is what comes back, is the datum.
UNITS=$(cat <<'EOF'
qwen35-08b|qwen35/Qwen3.5-0.8B-Q4_K_M.gguf|quant|1|smallest; proves the harness before anything expensive
qwen35-08b-f16|qwen35/Qwen3.5-0.8B-F16.gguf|f16|1|
llama32-3b|llama32-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf|quant|1|
llama32-3b-f16|llama32-3b/Llama-3.2-3B-Instruct-F16.gguf|f16|1|
ministral3-3b|ministral3-3b/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf|quant|1|
ministral3-3b-f16|ministral3-3b/Ministral-3-3B-Instruct-2512-F16.gguf|f16|1|
phi4mini|phi4mini/Phi-4-mini-instruct-Q4_K_M.gguf|quant|1|
phi4mini-f16|phi4mini/Phi-4-mini-instruct-F16.gguf|f16|1|
smolvlm2|smolvlm2/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf|quant|1|text rows only; the vision path is a separate item
qwen35-9b|qwen35/Qwen3.5-9B-Q4_K_M.gguf|quant|1|5.29+16.69 GiB resident: fits, but not with room
ministral3-8b|ministral3/Ministral-3-8B-Instruct-2512-Q4_K_M.gguf|quant|1|
gemma4-12b|gemma4/gemma-4-12b-it-Q4_K_M.gguf|quant|1|THE 12B+ QUESTION: 6.87+22.20 GiB. May refuse; the refusal is the datum
phi4-14b|phi4/phi-4-Q4_K_M.gguf|quant|0|no F16 staged; fp16 resident ~29 GiB beside the GGUF, does not fit
qwen36-27b|qwen36/Qwen3.6-27B-Q4_K_M.gguf|quant|0|-ub lever only; fp16 resident is ~54 GiB
gpt-oss-20b|gpt-oss-20b/gpt-oss-20b-mxfp4.gguf|moe|0|
deepseek-v2-lite|deepseek-v2-lite/DeepSeek-V2-Lite.Q4_K_M.gguf|moe|0|
qwen3-30b-a3b|qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf|moe|0|29 of 30.5 B are experts: cannot be held resident here
llama32-3b-qres|llama32-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf|qres|1|the unstacked recipe for the sub-4B class
ministral3-3b-qres|ministral3-3b/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf|qres|1|
phi4mini-qres|phi4mini/Phi-4-mini-instruct-Q4_K_M.gguf|qres|1|
smolvlm2-qres|smolvlm2/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf|qres|1|the one resolved -ub LOSS
qwen35-9b-qres|qwen35/Qwen3.5-9B-Q4_K_M.gguf|qres|1|the class edge: the one model whose -ub residue WINS
ministral3-8b-qres|ministral3/Ministral-3-8B-Instruct-2512-Q4_K_M.gguf|qres|1|second model of the 8-9 B class; the class rule must not turn on one model
gemma4-12b-qres|gemma4/gemma-4-12b-it-Q4_K_M.gguf|qres|1|the PARTIAL-residency case: places 73-74%, so the streamed remainder still pays the per-micro-batch dequant
qwen35-08b-dqc|qwen35/Qwen3.5-0.8B-Q4_K_M.gguf|dqc|1536|dense -ub mechanism control; proves the dqc arms cheaply
qwen35-9b-dqc|qwen35/Qwen3.5-9B-Q4_K_M.gguf|dqc|15360|control on the largest measured -ub lever (1.424x); fp16 image 13184 MB
smolvlm2-dqc|smolvlm2/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf|dqc|4096|control on the one resolved -ub LOSS (0.941x)
llama32-3b-dqc|llama32-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf|dqc|6144|does the sub-4B class carry a negative residue? fp16 image 5232 MB
ministral3-3b-dqc|ministral3-3b/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf|dqc|6656|fp16 image 5610 MB
phi4mini-dqc|phi4mini/Phi-4-mini-instruct-Q4_K_M.gguf|dqc|7168|fp16 image 6000 MB
EOF
)

if [ "${1:-}" = --list ]; then
  printf '%-20s %-6s %-4s %s\n' KEY CLASS ARMS NOTE
  while IFS='|' read -r key gguf class fits note; do
    [ -n "$key" ] || continue
    case $class in
      quant) n=$((2 + fits + CPUARM)) ;;
      qres)  n=$((2 + CPUARM)) ;;
      f16)   n=2 ;;
      moe)   n=$((4 + CPUARM)) ;;
      dqc)   n=4 ;;
    esac
    printf '%-20s %-6s %-4s %s\n' "$key" "$class" "$n" "$note"
  done <<< "$UNITS"
  echo
  echo "MODE=$MODE  CPUARM=$CPUARM  TESTS='$TESTS'  OUTD=$OUTD"
  echo "Cost per arm ~= (sum of -p) x (reps + 1 warmup) tokens of prefill, plus one discarded"
  echo "warmup process and one model load. Divide by the model's pp2048 t/s for wall time."
  echo "Then MULTIPLY BY PASSES (default 3): one process per arm does not settle a sign on this"
  echo "board -- the same two-arm unit read 0.948x and then 1.107x with identical placement."
  echo "PASSES=$(printf %s "${PASSES:-3}")"
  exit 0
fi

[ $# -ge 1 ] || { echo "usage: $0 MODEL_KEY... | --list   (see --list for keys)" >&2; exit 2; }
mkdir -p "$OUTD"
for want in "$@"; do
  line=$(grep "^$want|" <<< "$UNITS") || { echo "unknown key: $want" >&2; exit 2; }
  IFS='|' read -r key gguf class fits note <<< "$line"
  path="$MODELS/$gguf"
  [ -f "$path" ] || { echo "MISSING, SKIPPED: $key -> $path" | tee -a "$OUTD/skipped.md"; continue; }
  case $class in
    quant) ARMS=$(arms_quant "$fits") ;;
    qres)  ARMS=$(arms_qres) ;;
    f16)   ARMS=$(arms_f16) ;;
    moe)   ARMS=$(arms_moe) ;;
    dqc)   ARMS=$(arms_dqc "$fits") ;;   # for dqc rows the 4th field is the cache budget in MB
  esac
  export ARMS
  export OUT="$OUTD/$key.md"
  export ERRD="$OUTD/err"
  : > "$OUT"
  {
    echo "<!-- $key  class=$class  fp16-resident-fits=$fits  MODE=$MODE  TESTS='$TESTS'"
    echo "     gguf=$path ($(stat -c %s "$path") bytes)"
    [ -n "$note" ] && echo "     note: $note"
    echo "-->"
  } >> "$OUT"
  echo "=== $key ($class, $(wc -l <<< "$ARMS") arms) -> $OUT"
  bash "$GEN" "$path" "$key"
done
