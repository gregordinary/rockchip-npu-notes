#!/bin/bash
# qres512-9b-driver.sh -- the one class edge the unstacked-residency question never measured.
#
# `ROCKET_QUANT_RESIDENT=auto` at the DEFAULT -ub, against stock, on qwen35-9b. Every other
# unit run this way is sub-4B, and every sub-4B model carries a NEGATIVE non-dequant residue
# (0.789-0.864) -- so on all of them dropping `-ub 2048` from the recipe gives something back.
# The 9B is the one measured model whose residue is positive (1.038x), which makes it the case
# where the inherited reasoning says unstacking should LOSE. Same protocol as the three sub-4B
# qres512 units so the rows are comparable: three rotated interleaved passes, ratios paired
# within a pass, one reset per arm.
#
# It also carries the per-process readout's first rows on a second model, and the
# stock-vs-resident pair is the memory counters' dynamic-range contrast: the resident arm
# holds ~13 GB of fp16 on
# the NPU that the stock arm re-dequantises per micro-batch, so l3ref_pki and memacc_pki have to
# separate the arms or those columns are not reading anything.
set -u
HERE=${HERE:-$(cd "$(dirname "$0")" \&\& pwd)}
export SO=$HOME/npu/ggml-rocket/build-dl/libggml-rocket.so
export BIN=$HOME/npu/llama.cpp/build/bin/llama-bench
export TESTS="-p 2048 -n 0 -r 3"
export ARMS=$'stock||\nqres512|ROCKET_QUANT_RESIDENT=auto|'
export ERRD=$HERE/tuning-matrix/err
export OUT=$HERE/tuning-matrix/qwen35-9b-qres512.md
echo "== audit before qwen35-9b-qres512 ($(date +%T)) =="
ps -eo pcpu,pid,args --sort=-pcpu | head -5
echo -n "governors: "
for c in 0 4 6; do
  printf '%s ' "$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor)"
done; echo
: > "$OUT"
bash $HERE/bench-llm.sh "$HERE/qwen35/Qwen3.5-9B-Q4_K_M.gguf" qwen35-9b-qres512
echo "== rc=$? ($(date +%T)) =="
echo "[$(date +%T)] ALL-DONE"
