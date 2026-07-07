#!/bin/bash
# bench-llm.sh MODEL LABEL [extra llama-bench args...]
# One pass = CPU baseline + NPU, warm, covering prefill / decode / interactive.
#   prefill:     -p 512,1024,2048    (prompt-processing curve; the NPU's job)
#   decode:      -n 64               (steady-state generation; CPU-bound either way)
#   interactive: -pg 2048,128        (RAG/summarize turn: long prompt, short reply -> NPU wins;
#                                      validates the TTFT + stream decomposition)
# Markdown rows appended to $OUT. Warm run only (a discarded warmup spins the NPU clock up).
# Paths are env-overridable: SO / BIN / OUT (defaults are $HOME-relative; override for your layout).
set -u
SO=${SO:-$HOME/ggml-rocket/build-dl/libggml-rocket.so}
BIN=${BIN:-$HOME/llama.cpp/build/bin/llama-bench}
OUT=${OUT:-./bench_results.md}

MODEL="$1"; LABEL="$2"; shift 2
EXTRA="$*"                                    # e.g. "-b 2048 -ub 2048" for quant streaming
TESTS="-p 512,1024,2048 -n 64 -pg 2048,128 -r 2"   # prefill curve + clean decode + 1 combined turn (validates the TTFT+gen decomposition)

run() { # $1 = npu|cpu
  local mode="$1" envs=""
  [ "$mode" = npu ] && envs="GGML_BACKEND_PATH=$SO ROCKET_KACC=1"
  # discarded warmup: spin the NPU clock off idle before the measured run
  env $envs $BIN -m "$MODEL" -p 512 -n 8 -r 1 $EXTRA >/dev/null 2>&1
  echo "### $LABEL  [$mode]  $(date +%T)  clk=$(sudo cat /sys/kernel/debug/clk/clk_summary 2>/dev/null | awk '/scmi_clk_npu/{printf "%d MHz",$5/1e6; exit}')" | tee -a "$OUT"
  env $envs $BIN -m "$MODEL" $TESTS $EXTRA -o md 2>/dev/null | tee -a "$OUT"
  echo | tee -a "$OUT"
}
echo "== $LABEL  $(date) ==" | tee -a "$OUT"
run cpu
run npu
