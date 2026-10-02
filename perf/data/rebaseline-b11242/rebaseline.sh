#!/bin/bash
# Re-baseline of the quantized README rows at llama.cpp b11242: CPU repack off / on, and the NPU
# arm with --repack 0. Runs as ROOT (launched once through sudo -S) so bench-llm.sh's own sudo
# calls (drop_caches, compact_memory, perf, pagemap, the clk read) need no password.
# Usage: rebaseline.sh smoke|full
set -u
D=/path/to/data/rebase/rebaseline; mkdir -p "$D"
MODE=${1:-full}
exec 9>$HOME/npu/.npu.lock; flock 9
CF=/sys/devices/system/cpu/cpufreq
declare -A GOV
for p in $CF/policy*; do GOV[$p]=$(cat $p/scaling_governor); echo performance > $p/scaling_governor; done
echo "=== $(date -u +%T) $MODE  governor pinned: $(for p in $CF/policy*; do printf '%s:%s ' $(basename $p) $(cat $p/scaling_governor); done)  npu_clk=$(cat /sys/module/rocket/parameters/rocket_npu_clk_hz)"
export SO=$HOME/npu/base/ggml-rocket/build-b11242/libggml-rocket.so
export BIN=/path/to/data/rebase/llama.cpp-b11242/build/bin/llama-bench
ARMS=$'cpu-rp0||-b 2048 -ub 2048 --repack 0\ncpu-rp1||-b 2048 -ub 2048 --repack 1\nnpu||-b 2048 -ub 2048 --repack 0'
if [ "$MODE" = smoke ]; then
  LIST="smoke-llama32-3b-q4|/path/to/data/llama32-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf|1"
  export TESTS='-p 512 -n 0 -r 1'
else
  LIST=$'deepseek-v2-lite|/path/to/data/deepseek-v2-lite/DeepSeek-V2-Lite.Q4_K_M.gguf|3\ngpt-oss-20b|/path/to/data/gpt-oss-20b/gpt-oss-20b-mxfp4.gguf|3\nphi4-14b|/path/to/data/phi4/phi-4-Q4_K_M.gguf|3\nqwen36-27b|/path/to/data/qwen36/Qwen3.6-27B-Q4_K_M.gguf|2'
  export TESTS='-p 2048 -n 0 -r 1'
fi
while IFS='|' read -r label gguf passes; do
  [ -n "$label" ] || continue
  echo "=== $(date -u +%T) $label start"
  OUT=$D/$label.md ARMS="$ARMS" PASSES=$passes bash $D/bench-llm.sh "$gguf" "$label" > $D/$label.console 2>&1
  echo "=== $(date -u +%T) $label rc=$?"
  grep -A12 "^#### summary" $D/$label.md
done <<< "$LIST"
for p in "${!GOV[@]}"; do echo "${GOV[$p]}" > $p/scaling_governor; done
chown -R debian:debian "$D"
echo "=== $(date -u +%T) DONE  governor restored: $(cat $CF/policy0/scaling_governor)"
