#!/bin/bash
# sherpa-onnx CPU baseline, both Parakeet TDT int8 exports.
set -u
S=/path/to/data/stt-short/sherpa
O=/path/to/data/stt-short/run5; mkdir -p $O
PY=$S/venv/bin/python
{
echo "start $(date -u) gov=$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_governor)"
for m in v2 v3; do
  taskset -c 4-7 $PY $S/sherpa_bench.py $S/sherpa-onnx-nemo-parakeet-tdt-0.6b-$m-int8 $S/clips-wav 4 3 > $O/$m.jsonl 2> $O/$m.err
  echo "$(date -u +%T) $m rc=$? lines=$(wc -l < $O/$m.jsonl)"
done
echo "done $(date -u)"
} >> $O/campaign.log 2>&1
