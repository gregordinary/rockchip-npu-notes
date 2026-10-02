#!/bin/bash
# LLM crossover re-check: Llama-3.2-3B F16, NPU at floor 4 against CPU.
set -u
L=/path/to/data/rebase/llama.cpp-b11242
LM=/path/to/data/llama32-3b/Llama-3.2-3B-Instruct-F16.gguf
O=${O:-/path/to/data/stt-short/run4}
mkdir -p $O
export LD_LIBRARY_PATH=$L/build/bin GGML_BACKEND_PATH=/path/to/data/rebase/gr2-llama/libggml-rocket.so
lb() { # tag env...
  local tag=$1; shift
  env "$@" taskset -c 4-7 $L/build/bin/llama-bench -m $LM -p 16,32,64,128 -n 0 -r 3 -t 4 -o jsonl > $O/$tag.jsonl 2> $O/$tag.err
  echo "$(date -u +%T) $tag rc=$? $(grep -o '"n_prompt": [0-9]*\|"avg_ts": [0-9.]*' $O/$tag.jsonl | paste -sd' ')"
}
{
echo "start $(date -u)"
env ROCKET_MIN_M=4 taskset -c 4-7 $L/build/bin/llama-bench -m $LM -p 16 -n 0 -r 1 -t 4 > /dev/null 2>&1   # warm-up
for r in 1 2; do
  if (( r % 2 )); then lb npu.r$r ROCKET_MIN_M=4; lb cpu.r$r ROCKET_MIN_M=1000000 ROCKET_FLASH_ATTN=0
  else lb cpu.r$r ROCKET_MIN_M=1000000 ROCKET_FLASH_ATTN=0; lb npu.r$r ROCKET_MIN_M=4; fi
done
echo "done $(date -u)"
} >> $O/campaign.log 2>&1
