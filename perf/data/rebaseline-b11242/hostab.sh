#!/bin/bash
# Why do the NPU arms read above the historical campaign? Three NPU arms, same ggml-rocket source
# (ggml-rocket as of 2026-09-28; gr2-* are built from a tree byte-identical to it), 3 passes rotated, memory reset and a
# discarded warm-up before every arm. Runs as ROOT, queues on the board lock behind the campaign.
#   old-pin : llama.cpp b10558 (REPACK=OFF build), governor performance
#   new-pin : llama.cpp b11242, --repack 0,          governor performance
#   new-ond : llama.cpp b11242, --repack 0,          governor ondemand
set -u
D=/path/to/data/rebase/rebaseline; O=$D/hostab.md; E=$D/hostab-err; mkdir -p $E
exec 9>$HOME/npu/.npu.lock; flock 9
CF=/sys/devices/system/cpu/cpufreq
declare -A GOV; for p in $CF/policy*; do GOV[$p]=$(cat $p/scaling_governor); done
gov() { for p in $CF/policy*; do echo "$1" > $p/scaling_governor; done; }
reset_mem() { sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory; }
OLD_BIN=$HOME/npu/llama.cpp/build/bin/llama-bench; OLD_SO=/path/to/data/rebase/gr2-b10558/libggml-rocket.so
NEW_BIN=/path/to/data/rebase/llama.cpp-b11242/build/bin/llama-bench; NEW_SO=/path/to/data/rebase/gr2-llama/libggml-rocket.so
ARMS=$'old-pin|performance|'"$OLD_BIN"'|'"$OLD_SO"$'|\nnew-pin|performance|'"$NEW_BIN"'|'"$NEW_SO"$'|--repack 0\nnew-ond|ondemand|'"$NEW_BIN"'|'"$NEW_SO"'|--repack 0'
echo "== hostab $(date -u)" | tee -a $O
for m in "deepseek-v2-lite|/path/to/data/deepseek-v2-lite/DeepSeek-V2-Lite.Q4_K_M.gguf" "phi4-14b|/path/to/data/phi4/phi-4-Q4_K_M.gguf"; do
  IFS='|' read -r ml gguf <<< "$m"
  for pass in 1 2 3; do
    ordered=$(awk -v r="$(( (pass - 1) % 3 ))" 'NF{a[++n]=$0} END{for(i=1;i<=n;i++) print a[((i-1+r)%n)+1]}' <<< "$ARMS")
    while IFS='|' read -r label g bin so args; do
      [ -n "$label" ] || continue
      gov "$g"; reset_mem
      GGML_BACKEND_PATH=$so $bin -m "$gguf" -p 512 -n 8 -r 1 -b 2048 -ub 2048 $args > /dev/null 2>&1
      GGML_BACKEND_PATH=$so ROCKET_LOG_STDERR=1 $bin -m "$gguf" -p 2048 -n 0 -r 1 -b 2048 -ub 2048 $args -o md \
        > $E/$ml.$label.p$pass.md 2> $E/$ml.$label.p$pass.err
      rc=$?
      grep -q "loaded ROCKET backend" $E/$ml.$label.p$pass.err || rc="$rc,NO-ROCKET"
      v=$(awk -F'|' '$(NF-2) ~ /pp2048/ {v=$(NF-1); sub(/±.*/,"",v); gsub(/ /,"",v); print v}' $E/$ml.$label.p$pass.md)
      x=$(grep -ho "experts exercised: [0-9]* resident\|[0-9]* streamed" $E/$ml.$label.p$pass.err | tr '\n' ' ')
      printf '<!--HAB %s\t%s\t%s\t%s\trc=%s\tgov=%s\t%s-->\n' "$ml" "$pass" "$label" "$v" "$rc" "$(cat $CF/policy4/scaling_governor)" "$x" | tee -a $O
    done <<< "$ordered"
  done
done
for p in "${!GOV[@]}"; do echo "${GOV[$p]}" > $p/scaling_governor; done
chown -R debian:debian $O $E
echo "== DONE $(date -u)  governor restored: $(cat $CF/policy0/scaling_governor)" | tee -a $O
