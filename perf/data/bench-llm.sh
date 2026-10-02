#!/bin/bash
# bench-llm.sh MODEL LABEL [extra llama-bench args...]
# One pass = CPU baseline + NPU, warm, covering prefill / decode / interactive.
#   prefill:     -p 512,1024,2048    (prompt-processing curve; the NPU's job)
#   decode:      -n 64               (steady-state generation; CPU-bound either way)
#   interactive: -pg 2048,128        (RAG/summarize turn: long prompt, short reply -> NPU wins;
#                                      validates the TTFT + stream decomposition)
# Markdown rows appended to $OUT. Warm run only (a discarded warmup spins the NPU clock up).
# Paths are env-overridable: SO / BIN / OUT (defaults are $HOME-relative; override for your layout).
#
# THE FLAG AXIS. Set ARMS and the same pass sweeps FLAGS instead of cpu-vs-npu: every arm on the
# NPU, so what a row reports is stock-default against recommended rather than against the CPU.
# That is the pairing the tuning matrix needs -- a user's question is "what does following the
# guide buy me", and the CPU arm cannot answer it.
#
#   ARMS='label|ENV=V ...|llama-bench args'          one arm per line, '|' between the fields.
#
# The separator is '|' and NOT a tab, which was the first thing this got wrong: a tab is IFS
# WHITESPACE, so `read` collapses a run of them into one delimiter and an EMPTY middle field
# disappears -- an arm written `ub2048<TAB><TAB>-b 2048 -ub 2048` parses as env='-b 2048 -ub
# 2048' with no args, which sets a nonsense variable and silently benchmarks the stock config
# under the tuned label. Empty fields are the norm here, so the delimiter cannot be whitespace.
#
# An EMPTY env field and an empty args field is the stock arm, and that is the point: stock is
# the absence of settings, so it must be spelled as an absence rather than as a flag that
# happens to match a default. The label `cpu`, or any `cpu-*` label, is special-cased to drop
# GGML_BACKEND_PATH, so the absolute reference can ride along in the same pass at the same clock.
# The `cpu-*` form is what lets two CPU arms share a pass, e.g. `cpu-rp1` and `cpu-rp0` for the
# host's weight repack on and off (`--repack 1|0`).
#
# Every arm runs under ROCKET_LOG_STDERR=1 and keeps its stderr, because the mode/budget/OUTCOME
# lines are what make a row self-documenting.
#
# PASSIVE PREDICTORS. Every timed arm also emits a <!--PRED pass arm k=v ...--> line, read once at
# the start of the timed run: MemAvailable, AnonHugePages/Hugepagesize, and the per-order free-page
# vector from /proc/buddyinfo. They cost two file reads and no process, and they are there so that
# the per-process spread can be attacked from rows already taken rather than from a campaign run
# later. PRED_SAMPLE=1 adds a per-thread last-CPU histogram, which is NOT free -- see pred_sample.
# These are RECORDED, NOT PREDICTIVE: over 177 joined rows they stay flat to <1.2% across 8-13%
# t/s swings, which is why the readout below exists. The outcome half is load-bearing: a residency
# arm that silently placed nothing produces the same rows and the same t/s as one that placed
# everything and gained nothing, and only the teardown split tells them apart.
#
# THE GOVERNOR, AND WHETHER THE BACKEND LOADED. The script refuses to run while any cpufreq policy
# is unpinned (see pinned_or_refuse), every PRED line records each policy's governor and floor, and
# an arm fails without the ROCKET registration line in its stderr (the `cpu` arm fails WITH one).
#
# THE PER-PROCESS READOUT. Every timed arm also emits a <!--RO pass arm k=v ...--> line covering
# what the board snapshot above cannot see -- where the work RAN (per-cluster instructions, per-CPU
# jiffies) and where its pages LANDED (cache colour, physical contiguity). See the block above
# ro_line() for the mechanism and ro-pagemap.py for the placement columns; RO=0 turns all of it
# off. Read `pmu_enabled` and `pfn_zero_frac` before quoting any of it: both failure modes are
# silent and produce a full, plausible line.
set -u
SO=${SO:-$HOME/ggml-rocket/build-dl/libggml-rocket.so}
BIN=${BIN:-$HOME/llama.cpp/build/bin/llama-bench}
OUT=${OUT:-./bench_results.md}
ERRD=${ERRD:-$(dirname "$OUT")/err}
ARMS=${ARMS:-}
# Interleaved repeats per arm (ARMS mode only). One process per arm does not settle a sign on this
# board -- see the reset_mem comment below -- so the default is three, and the cost of a unit is
# three times its arm cost.
PASSES=${PASSES:-3}

MODEL="$1"; LABEL="$2"; shift 2
# LABEL names every file an arm writes. A path separator in it points the stderr redirect at a
# directory that does not exist, and the arm then never runs.
case "$LABEL" in */*) echo "bench-llm.sh: LABEL '$LABEL' names files and cannot contain '/'" >&2; exit 2 ;; esac
EXTRA="$*"                                    # e.g. "-b 2048 -ub 2048" for quant streaming
# TESTS is overridable because the flag axis multiplies the cost by the number of arms: the
# headline "what does tuning buy" table needs only the pp2048 row, and paying for the decode and
# interactive rows on every arm is what turns a model into an hour.
TESTS=${TESTS:-"-p 512,1024,2048 -n 64 -pg 2048,128 -r 2"}

run() { # $1 = npu|cpu
  local mode="$1" envs=""
  [ "$mode" = npu ] && envs="GGML_BACKEND_PATH=$SO ROCKET_KACC=1"
  mkdir -p "$ERRD"
  reset_mem
  # discarded warmup: spin the NPU clock off idle before the measured run
  env $envs $BIN -m "$MODEL" -p 512 -n 8 -r 1 $EXTRA >/dev/null 2>&1
  echo "### $LABEL  [$mode]  $(date +%T)  clk=$(sudo cat /sys/kernel/debug/clk/clk_summary 2>/dev/null | awk '/scmi_clk_npu/{printf "%d MHz",$5/1e6; exit}')  cpufreq=$(cpufreq_state)" | tee -a "$OUT"
  env $envs $BIN -m "$MODEL" $TESTS $EXTRA -o md 2>"$ERRD/$LABEL.$mode.err" | tee -a "$OUT"
  registered_ok "$mode" "$ERRD/$LABEL.$mode.err" || \
    echo "    ARM FAILED: the [$mode] run's ROCKET registration line is $([ "$mode" = cpu ] && echo present || echo absent) -- see $ERRD/$LABEL.$mode.err" | tee -a "$OUT"
  echo | tee -a "$OUT"
}
clk() { sudo cat /sys/kernel/debug/clk/clk_summary 2>/dev/null | awk '/scmi_clk_npu/{printf "%d MHz",$5/1e6; exit}'; }

# Reset the board's memory state before every arm. The board drifts a LEVEL: one unchanged
# configuration read 109.59 t/s, then 118.23 and 118.16 after a drop_caches + compact_memory, then
# 116.78 with no further reset -- an 8% step that had persisted across ~20 processes and 40 minutes
# [HW sweep 2026-08-28, RK1, Qwen3.5-0.8B-Q4_K, 600 MHz pinned]. What degrades the board is NOT
# pinned (repeating the cell did not do it, nor did one 3B-F16 residency run), so the two halves of
# the reset could not be separated and both are applied as a cure. Cost is a cold model load per
# arm, paid by the warm-up below, outside the timed region.
#
# THE RESET IS NOT ENOUGH ON ITS OWN, which is why PASSES exists below: with it in place, one
# process per arm still does not settle a sign. The same two-arm unit run twice read 123.46/117.10
# (0.948x) and then 117.29/129.85 (1.107x) -- identical placement in all four, so the inversion is
# process-to-process spread and not the knob. Anything this axis measures is 2-10%, which is inside
# that. Repeats interleaved across arms are the instrument; the reset only removes the slow level
# drift underneath them.
reset_mem() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1
              echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null 2>&1; }

# THE GOVERNOR IS PART OF THE OPERATING POINT, not hygiene. An offloading process blocks with its
# threads off the run queue, so a load-sampling governor parks the big cores at scaling_min_freq
# and the host half of the work runs there, while a CPU-only arm keeps every core busy and fast.
# Unpinned, an NPU arm read up to 3.2x slow against a CPU arm that lost nothing [HW sweep, RK1,
# perf/cpu-governor-and-offload.md]. A policy counts as pinned when its governor is `performance`
# or its floor equals its ceiling. The floor is the size of the effect, so it is recorded too.
# CPUFREQ_DIR is overridable so the dry run can hand the script a pinned and an unpinned tree.
CPUFREQ_DIR=${CPUFREQ_DIR:-/sys/devices/system/cpu/cpufreq}
cpufreq_state() {  # one token: policyN:governor:min-max, comma-separated
  local p s=""
  for p in "$CPUFREQ_DIR"/policy*; do
    [ -r "$p/scaling_governor" ] || continue
    s="$s${s:+,}$(basename "$p"):$(cat "$p/scaling_governor"):$(cat "$p/scaling_min_freq")-$(cat "$p/scaling_max_freq")"
  done
  echo "${s:-none}"
}
unpinned_policies() {  # the policies neither under `performance` nor with the floor at the ceiling
  local p out=""
  for p in "$CPUFREQ_DIR"/policy*; do
    [ -r "$p/scaling_governor" ] || continue
    [ "$(cat "$p/scaling_governor")" = performance ] && continue
    [ "$(cat "$p/scaling_min_freq")" = "$(cat "$p/scaling_max_freq")" ] && continue
    out="$out${out:+ }$(basename "$p")"
  done
  echo "$out"
}
# Refuse an unpinned board unless ALLOW_UNPINNED=1 says the run is meant to be unpinned, as every
# campaign number in the tuning matrix is. Either way the state lands on each PRED line.
pinned_or_refuse() {
  local unp; unp=$(unpinned_policies)
  [ -z "$unp" ] && return 0
  if [ "${ALLOW_UNPINNED:-0}" = 1 ]; then
    echo "bench-llm.sh: running UNPINNED ($unp) because ALLOW_UNPINNED=1: $(cpufreq_state)" | tee -a "$OUT"
    return 0
  fi
  echo "bench-llm.sh: refusing to run, cpufreq policies not pinned ($unp): $(cpufreq_state)" >&2
  echo "  pin them (governor performance, or scaling_min_freq = scaling_max_freq)," >&2
  echo "  or set ALLOW_UNPINNED=1 for a deliberately unpinned run" >&2
  exit 2
}

# WHETHER THE BACKEND LOADED. ggml prints `load_backend: loaded ROCKET backend from <path>` when
# GGML_BACKEND_PATH resolves, and a wrong path prints only a `failed to load` line. So an NPU arm
# without the line ran on the CPU under an NPU label, and a `cpu` arm with it ran the NPU under a
# CPU label (an inherited GGML_BACKEND_PATH does that). Either one's numbers are kept out of DATA.
is_cpu_label() { case "$1" in cpu|cpu-*) return 0 ;; esac; return 1; }
registered_ok() {  # $1 = arm label, $2 = the arm's stderr file
  if is_cpu_label "$1"; then ! grep -q "loaded ROCKET backend" "$2" 2>/dev/null
  else grep -q "loaded ROCKET backend" "$2" 2>/dev/null; fi
}

# PASSIVE PREDICTORS for the per-process spread, recorded at the START of the timed run (after the
# warm-up, so this arm's residency ingest and its page faults are already paid). One <!--PRED-->
# line per (pass, arm) beside the <!--DATA--> rows, so every future matrix row carries its own
# candidate predictors and the ~11% per-process spread can be attacked RETROSPECTIVELY rather than
# with a dedicated campaign. Nothing here is known to predict anything yet; these are the three
# cheapest suspects left standing after thermals, the governor, the NPU clock, -r, ROCKET_KACC and
# within-process variance were ruled out [HW sweep 2026-08-28, RK1].
#
#   memavail_kb    what the MoE and residency pre-flights resolve their budget from, once per
#                  PROCESS -- so it is an input to placement, not only a symptom
#   anonhuge_kb    THP actually backing anonymous memory, with Hugepagesize beside it. READS 0 ON
#                  THE RK1 BY CONSTRUCTION, and that is not a broken instrument: its 7.2.0-1-arm64
#                  kernel has `# CONFIG_TRANSPARENT_HUGEPAGE is not set` (/proc/config.gz), there
#                  is no /sys/kernel/mm/transparent_hugepage, and /proc/meminfo carries no
#                  AnonHugePages line at all [verified on the board 2026-08-28]. So the 4K-vs-2M
#                  hypothesis is dead for the RK1's spread. It is still recorded because it costs
#                  nothing and a different board or kernel would make it live -- a column of zeros
#                  here is a POSITIVE statement that THP is not the mechanism, not a missing read
#   buddy          free-page counts per order, summed over zones, order 0 first. Fragmentation is
#                  the standing suspect: compact_memory was half of what cured the level drift, and
#                  the halves could not be separated because the degraded state would not come back
#
# The WHOLE buddyinfo vector is recorded rather than a chosen high-order cut, on purpose: a later
# analysis picks its own order, and a kernel with a different MAX_ORDER cannot silently shift a
# column under a fixed one.
predictors() {  # $1 = pass, $2 = arm label
  local pinned=1; [ -n "$(unpinned_policies)" ] && pinned=0
  awk -v p="$1" -v l="$2" -v cf="$(cpufreq_state)" -v pin="$pinned" '
    FILENAME ~ /meminfo/ {
      if ($1 == "MemAvailable:")  m = $2
      if ($1 == "AnonHugePages:") a = $2
      if ($1 == "Hugepagesize:")  h = $2
    }
    FILENAME ~ /buddyinfo/ && /zone/ {
      for (i = 5; i <= NF; i++) f[i-4] += $i
      if (NF - 4 > n) n = NF - 4
    }
    END {
      printf "<!--PRED %s\t%s\tmemavail_kb=%d\tanonhuge_kb=%d\thugepagesz_kb=%d\tbuddy=",
             p, l, m+0, a+0, h+0
      for (i = 1; i <= n; i++) printf "%s%d", (i > 1 ? "," : ""), f[i] + 0
      printf "\tpinned=%d\tcpufreq=%s", pin, cf
      printf "%s", "-->\n"
    }' /proc/meminfo /proc/buddyinfo 2>/dev/null
}

# LAST-CPU PER THREAD IS NOT FREE, which is why it is opt-in (PRED_SAMPLE=1) and not part of the
# line above. It can only be sampled DURING the run (/proc/<pid>/task/*/stat field 39), so it needs
# a sampler process running alongside the timed one -- an added tenant on a board whose per-process
# spread is the very thing being measured. Turning it on changes the workload it is measuring. That
# is the trade, and it is why the default line stays to one-shot reads of two files.
#
# pgrep -x, matching the process NAME: `pgrep -f llama-bench` matches this script's own command
# line, and a watcher that matches itself never exits.
#
# The wait for llama-bench to APPEAR is bounded at 100 x 0.2 s. If it never does -- a failed arm,
# or a BIN that is not named llama-bench -- the sampler writes an empty file, the [ -s ] guard
# below drops the line, and the arm pays 20 s once. Verified against a stub 2026-08-28: it
# terminates and emits nothing rather than hanging. Irrelevant beside a real 400 s prefill; it is
# the reason not to leave PRED_SAMPLE=1 on for a matrix of short cells.
pred_sample() {  # $1 = output file. Backgrounded; exits when llama-bench does.
  local p= i=0
  while [ "$i" -lt 100 ]; do
    p=$(pgrep -x llama-bench | head -1)
    [ -n "$p" ] && break
    i=$((i + 1)); sleep 0.2
  done
  [ -n "$p" ] || { : > "$1"; return; }
  while [ -d "/proc/$p" ]; do
    awk '{ print $39 }' /proc/"$p"/task/*/stat 2>/dev/null
    sleep 0.5
  done | sort -n | uniq -c | sort -rn | awk '{ printf "%s%s:%s", (NR > 1 ? "," : ""), $2, $1 }' > "$1"
}

# THE PER-PROCESS READOUT. The predictors above are a snapshot of the BOARD, and as
# predictors of the per-process wall spread they are closed: over 177 joined rows plus a 24-arm
# re-run, MemAvailable and the whole buddyinfo vector stay flat to <1.2% across 8-13% t/s swings.
# What that leaves is what the allocator and the scheduler hand THIS PROCESS, which no snapshot of
# the board can see. So this reads the process, and it reads it DURING the timed run:
#
#   RO_PMU=1   system-wide `perf stat` across exactly the timed region, on BOTH cluster PMUs.
#              Per-cluster inst_retired is the thread-placement readout -- an A55 is roughly a
#              third of an A76 here, so a run whose threads drift onto the little cluster is
#              slower for a reason that has nothing to do with the knob under test. The A76
#              memory events (l2d/l3d refill, mem_access, dtlb_walk), normalised PER INSTRUCTION
#              so they do not merely restate how much work ran, are the cache-congruence readout.
#              System-wide and not per-task deliberately: it leaves the workload's own command
#              line, user and environment untouched, and the board is audited idle anyway. Its
#              own cost is a counter program plus one wakeup -- it sleeps for the whole run.
#   RO_PM=1    one bounded ro-pagemap.py sample once the process's RSS has stopped growing: where
#              its pages actually landed, as cache colour and physical contiguity. See that file
#              for what each column means and for the positive control it has to pass first.
#   busy=      per-CPU jiffies delta across the timed region, from two /proc/stat reads. Free, and
#              it is the cross-check on the PMU's cluster split: two instruments disagreeing about
#              where the work ran is a fact about the instruments.
#
# ALL OF IT IS BEST-EFFORT AND NONE OF IT CAN FAIL AN ARM. Every piece is backgrounded or guarded,
# `sudo -n` so a missing credential fails fast instead of hanging a detached campaign, and the
# timed command's own rc is captured before any of this is read. A readout that can break the
# measurement it annotates is worse than no readout.
#
# READ pmu_enabled BEFORE QUOTING A COUNTER. The A76 list is sized to its six programmable
# counters exactly; add an event and perf multiplexes, which scales every count silently. Below
# 99 means the numbers are estimates.
#
# AND pmu_cpu_s IS NOT THE ARM'S WALL. perf reports a counter's running time SUMMED OVER THE CPUS
# THE EVENT RAN ON, so a system-wide software event on this part reads ~8x the elapsed seconds and
# a cluster PMU event ~4x. It is a duration proxy and a correct RATIO between arms; the arm's
# actual elapsed time is `wall_s`, taken from the shell around the timed command.
RO=${RO:-1}
RO_PMU=${RO_PMU:-1}
RO_PM=${RO_PM:-1}
A76=armv8_cortex_a76
A55=armv8_cortex_a55
RO_EVENTS=${RO_EVENTS:-"$A76/inst_retired/,$A76/cpu_cycles/,$A76/l2d_cache_refill/,$A76/l1d_cache_refill/,$A76/mem_access/,$A76/dtlb_walk/,$A76/l3d_cache_refill/,$A55/inst_retired/,$A55/cpu_cycles/,context-switches,cpu-migrations,page-faults"}

# Per-CPU busy jiffies (user+nice+system+irq+softirq), one line, no forks.
ro_cpu_snapshot() {
  local cpu u n s rest
  while read -r cpu u n s rest; do
    case $cpu in cpu[0-9]*) printf '%s ' "$((u + n + s))" ;; cpu) ;; *) break ;; esac
  done < /proc/stat
  echo
}

# Wait for the timed process, let it warm in, take ONE placement sample, exit. Polls
# /proc/PID/statm with the read builtin -- no process per poll, unlike pred_sample.
#
# THE DEADLINE IS LOAD-BEARING, and waiting for RSS to SETTLE alone is a silent no-op. Every arm
# is preceded by drop_caches, so the mmapped GGUF faults in off NVMe for the whole run and the
# resident size never stops growing: the first build of this waited for two consecutive stable
# reads, never got them, fell out of the loop only when the process EXITED, and wrote an empty
# file. It emitted `pagemap=absent` on a perfectly healthy arm -- an instrument that silently
# measures nothing looks exactly like one whose column is flat. So the wait ends at whichever
# comes first, settled or RO_PM_MAX polls, and the sample carries `ro_at_s`: which phase of the
# run it describes is part of the datum, not an assumption the reader has to make.
RO_PM_MAX=${RO_PM_MAX:-30}     # polls of 2 s before sampling regardless
ro_pagemap_sample() { # $1 = output file
  local p= i=0 prev=0 now=0 stable=0 t0=$SECONDS junk
  while [ "$i" -lt 200 ]; do p=$(pgrep -x llama-bench | head -1); [ -n "$p" ] && break; i=$((i+1)); sleep 0.2; done
  [ -n "$p" ] || { : > "$1"; return; }
  i=0
  while [ "$i" -lt "$RO_PM_MAX" ] && [ -d "/proc/$p" ]; do
    sleep 2; i=$((i+1))
    read -r junk now junk < /proc/"$p"/statm 2>/dev/null || break
    [ "${now:-0}" -gt 0 ] || break
    if [ "$prev" -gt 0 ] && [ $((now - prev)) -lt $((now / 50)) ]; then
      stable=$((stable+1)); [ "$stable" -ge 2 ] && break
    else stable=0; fi
    prev=$now
  done
  [ -d "/proc/$p" ] || { : > "$1"; return; }
  sudo -n python3 "$RO_PAGEMAP" "$p" \
       --label "at_s=$((SECONDS - t0)),rss_pg=$now,settled=$stable" > "$1" 2>/dev/null \
    || : > "$1"
}
RO_PAGEMAP=${RO_PAGEMAP:-$(dirname "${BASH_SOURCE[0]}")/ro-pagemap.py}

# One <!--RO pass arm k=v ...--> line per timed arm, beside <!--DATA--> and <!--PRED-->. Raw
# counts AND the derived ratios are both emitted: the ratios are what a spread is read from, and
# the raw counts are what lets a later analysis pick a different denominator without re-running
# the board. The memory events are normalised per THOUSAND A76 instructions -- an arm that simply
# ran more work would otherwise show more refills and read as worse placement.
ro_line() { # $1 pass, $2 arm, $3 cpu snapshot before, $4 after, $5 pmu csv, $6 pagemap, $7 wall s
  local pass="$1" label="$2" before="$3" after="$4" pmuf="$5" pmf="$6" wall="${7:-0}"
  printf '<!--RO %s\t%s\twall_s=%s\t' "$pass" "$label" "$wall"
  awk -v b="$before" -v a="$after" 'BEGIN{
    nb=split(b,B," "); na=split(a,A," "); n=(nb<na?nb:na); tot=0; lit=0; s=""
    for(i=1;i<=n;i++){d=A[i]-B[i]; if(d<0)d=0; s=s (i>1?",":"") d; tot+=d; if(i<=4) lit+=d}
    printf "busy=%s\tbusy_tot=%d\tbusy_little_share=%.4f\t", s, tot, (tot>0?lit/tot:0) }'
  if [ -s "$pmuf" ]; then
    awk -F, 'function key(e,  k){k=e; sub(/^armv8_cortex_/,"",k); sub(/\/$/,"",k);
                                 gsub(/\//,"_",k); gsub(/-/,"_",k); return k}
      BEGIN{minen=101; secs=0}
      $3 != "" && $1 !~ /^#/ { v = ($1 ~ /^[0-9]/) ? $1+0 : -1; c[key($3)] = v
                               if ($5+0 > 0 && $5+0 < minen) minen = $5+0
                               if ($4+0 > secs) secs = $4+0 }
      END{
        for (k in c) printf "%s=%d\t", k, c[k]
        i76=c["a76_inst_retired"]; i55=c["a55_inst_retired"]
        if (i76+i55 > 0) printf "a55_inst_share=%.4f\t", i55/(i76+i55)
        if (c["a76_cpu_cycles"] > 0) printf "a76_ipc=%.3f\t", i76/c["a76_cpu_cycles"]
        if (i76 > 0) printf "l2ref_pki=%.3f\tl3ref_pki=%.3f\tl1dref_pki=%.3f\tmemacc_pki=%.2f\tdtlbw_pki=%.4f\t",
              1000*c["a76_l2d_cache_refill"]/i76, 1000*c["a76_l3d_cache_refill"]/i76,
              1000*c["a76_l1d_cache_refill"]/i76, 1000*c["a76_mem_access"]/i76,
              1000*c["a76_dtlb_walk"]/i76
        printf "pmu_enabled=%.1f\tpmu_cpu_s=%.1f\t", (minen>100?0:minen), secs/1e9 }' "$pmuf"
  else printf 'pmu=absent\t'; fi
  if [ -s "$pmf" ]; then tr -d '\n' < "$pmf"; else printf 'pagemap=absent'; fi
  printf '%s' "-->"; echo
}

arm() { # $1 = label, $2 = env assignments, $3 = extra llama-bench args, $4 = pass number
  local label="$1" envs="$2" args="$3" pass="${4:-1}" tag
  tag="$LABEL.$label.p$pass"
  is_cpu_label "$label" || envs="GGML_BACKEND_PATH=$SO $envs"
  mkdir -p "$ERRD"
  reset_mem
  # The warmup spins the NPU clock off idle and pays this arm's one-time residency ingest
  # OUTSIDE the measured run. It used to double as a ROCKET_DEBUG PLACEMENT PROBE, because the
  # residency routes reported their budget and not their outcome -- so an arm that placed almost
  # nothing read exactly like one that placed everything and gained nothing. The probe is gone:
  # both routes now report the resident/streamed split at teardown, in the TIMED run, through
  # the driver channel that ROCKET_LOG_STDERR tees. That is strictly better than the probe was,
  # which could not match the timed run's MemAvailable at the moment the floor latched and so
  # could disagree with it. Read the arm's own [moe-int8] / [f16-resident] lines below.
  env $envs $BIN -m "$MODEL" -p 512 -n 8 -r 1 $args >/dev/null 2>&1
  echo "### $LABEL [$label] pass $pass  env='$2'  args='$args'  $(date +%T)  clk=$(clk)  MemAvail=$(awk '/MemAvailable/{print $2}' /proc/meminfo) kB" | tee -a "$OUT"
  predictors "$pass" "$label" | tee -a "$OUT"
  local spid=
  if [ "${PRED_SAMPLE:-0}" = 1 ]; then pred_sample "$ERRD/$tag.cpu" & spid=$!; fi
  # --- readout opens; nothing below may fail the arm
  local perfpid= ropid= cpu0= cpu1=
  local t_arm0=$SECONDS
  if [ "$RO" = 1 ]; then
    cpu0=$(ro_cpu_snapshot)
    if [ "$RO_PMU" = 1 ]; then
      # perf counts system-wide for exactly as long as its WORKLOAD lives, and the workload here
      # is a `cat` blocked on a fifo. Ending the region is then a write, not a signal. The signal
      # route was tried and is a trap: backgrounding `sudo perf ...` gives you sudo's pid, so the
      # SIGINT lands on sudo, perf never sees it, and the arm blocks in `wait` behind a `sleep`
      # with 24 hours to run -- after the timed run has already finished and printed its rows.
      rm -f "$ERRD/$tag.fifo"; mkfifo "$ERRD/$tag.fifo" 2>/dev/null
      sudo -n perf stat -a -x, -e "$RO_EVENTS" -o "$ERRD/$tag.pmu" -- cat "$ERRD/$tag.fifo" >/dev/null 2>&1 &
      perfpid=$!
    fi
    if [ "$RO_PM" = 1 ] && [ -r "$RO_PAGEMAP" ]; then ro_pagemap_sample "$ERRD/$tag.pm" & ropid=$!; fi
  fi
  env $envs ROCKET_LOG_STDERR=1 $BIN -m "$MODEL" $TESTS $args -o md 2>"$ERRD/$tag.err" | tee -a "$OUT" | tee "$ERRD/$tag.md" >/dev/null
  local rc=${PIPESTATUS[0]}
  if [ "$RO" = 1 ]; then
    cpu1=$(ro_cpu_snapshot)
    # perf prints its counts on SIGINT; the `sleep` it wraps has to go with it. Give it a
    # moment to flush, then read the file -- an unflushed -o file reads as no PMU line at all.
    if [ -n "$perfpid" ]; then
      echo end > "$ERRD/$tag.fifo" 2>/dev/null &
      local wpid=$!
      # Bounded: a readout must never be able to hang a detached campaign, so the wait has a
      # deadline and the arm goes on without a PMU line rather than stopping.
      local w=0
      while [ $w -lt 60 ] && kill -0 "$perfpid" 2>/dev/null; do sleep 0.5; w=$((w+1)); done
      kill -0 "$perfpid" 2>/dev/null && sudo -n pkill -x -INT perf >/dev/null 2>&1
      wait "$perfpid" 2>/dev/null; kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null
      rm -f "$ERRD/$tag.fifo"
    fi
    [ -n "$ropid" ] && wait "$ropid" 2>/dev/null
    ro_line "$pass" "$label" "$cpu0" "$cpu1" "$ERRD/$tag.pmu" "$ERRD/$tag.pm" \
            "$((SECONDS - t_arm0))" | tee -a "$OUT"
  fi
  if [ -n "$spid" ]; then
    wait "$spid" 2>/dev/null
    [ -s "$ERRD/$tag.cpu" ] && printf '<!--PRED %s\t%s\tlastcpu=%s-->\n' \
        "$pass" "$label" "$(cat "$ERRD/$tag.cpu")" | tee -a "$OUT"
  fi
  [ "$rc" -eq 0 ] || echo "    ARM FAILED rc=$rc -- see $ERRD/$tag.err" | tee -a "$OUT"
  local reg=1
  if ! registered_ok "$label" "$ERRD/$tag.err"; then
    reg=0
    if is_cpu_label "$label"; then
      echo "    ARM FAILED: the cpu arm loaded the ROCKET backend (GGML_BACKEND_PATH inherited?) -- no DATA rows" | tee -a "$OUT"
    else
      echo "    ARM FAILED: no 'loaded ROCKET backend' line in $ERRD/$tag.err, so it ran on the CPU -- no DATA rows" | tee -a "$OUT"
    fi
  fi
  # One machine-readable datum per (pass, arm, test), so the summary below is computed from the
  # numbers rather than re-parsed out of prose. Kept as an HTML comment: the file is markdown.
  [ "$reg" = 1 ] && awk -F'|' -v p="$pass" -v l="$label" 'NF>6 && $(NF-2) ~ /pp|tg/ {
        t=$(NF-2); v=$(NF-1); gsub(/ /,"",t); gsub(/ /,"",v); sub(/±.*/,"",v);
        if (v+0 > 0) printf "<!--DATA %s\t%s\t%s\t%s-->\n", p, l, t, v }' \
      "$ERRD/$tag.md" | tee -a "$OUT"
  grep -hE "residency pre-flight|resident on the NPU|streamed via|admission first declined|budget reached|experts exercised|resident budget|MM_ASYM|K-accum|dq-cache" \
       "$ERRD/$tag.err" 2>/dev/null | sort -u | sed 's/^/    /' | tee -a "$OUT"
  echo | tee -a "$OUT"
}

# Per-arm mean over the passes, and each arm's ratio against the FIRST arm listed (stock) computed
# PAIRWISE within a pass and then averaged -- not as a ratio of the two means. Pairing within a
# pass is what removes the level drift the two arms shared; a ratio of means does not, and on this
# board the drift is larger than the effect. The per-arm spread and the PER-PASS ratios are printed
# beside the mean because the mean alone cannot be read: a set that straddles 1.00 is a result about
# the instrument and not about the knob, and once averaged it looks identical to a settled one.
summarize() {
  local base; base=$(head -1 <<< "$ARMS" | cut -d'|' -f1)
  echo "#### summary: per-arm mean over $PASSES passes, ratios paired within a pass against [$base]" | tee -a "$OUT"
  grep -h '^<!--DATA' "$OUT" | sed 's/<!--DATA //; s/-->//' | awk -F'\t' -v base="$base" '
    { v[$1 SUBSEP $2 SUBSEP $3]=$4; if(!($2 in armseen)){armseen[$2]=1; arms[++na]=$2}
      if(!($3 in tseen)){tseen[$3]=1; tests[++nt]=$3}; if(!($1 in pseen)){pseen[$1]=1; ps[++np]=$1} }
    END {
      printf "| test | arm | mean t/s | n | min | max | paired ratio | per-pass ratios |\n"
      printf "|---|---|---:|---:|---:|---:|---:|---|\n"
      for (ti=1; ti<=nt; ti++) { t=tests[ti]
        for (ai=1; ai<=na; ai++) { a=arms[ai]; s=0; n=0; lo=0; hi=0; rs=0; rn=0; list=""
          for (pi=1; pi<=np; pi++) { p=ps[pi]; k=p SUBSEP a SUBSEP t
            if (k in v) { x=v[k]+0; s+=x; n++; if(lo==0||x<lo)lo=x; if(x>hi)hi=x
              bk=p SUBSEP base SUBSEP t
              if (bk in v && v[bk]+0>0) { r=x/(v[bk]+0); rs+=r; rn++
                list = list (list==""?"":" ") sprintf("%.3f", r) } } }
          if (n) printf "| %s | %s | %.2f | %d | %.2f | %.2f | %s | %s |\n", t, a, s/n, n, lo, hi,
                        (rn && a!=base) ? sprintf("%.3fx", rs/rn) : "--",
                        (a!=base) ? list : "--" } } }' | tee -a "$OUT"
  echo | tee -a "$OUT"
}

pinned_or_refuse
echo "== $LABEL  $(date)  cpufreq=$(cpufreq_state) ==" | tee -a "$OUT"
if [ -n "$ARMS" ]; then
  # PASSES interleaved repeats, the arm order ROTATED by one each pass. Rotation is what stops an
  # arm's position in the sequence from being confounded with the arm: run in a fixed order, the
  # arm that always goes last always inherits the most history. Three passes is the minimum that
  # can show a straddle; raise it for a unit whose effect is small against its spread.
  nA=$(grep -c . <<< "$ARMS")
  for pass in $(seq 1 "$PASSES"); do
    ordered=$(awk -v r="$(( (pass - 1) % nA ))" 'NF{a[++n]=$0}
                  END{for(i=1;i<=n;i++) print a[((i-1+r)%n)+1]}' <<< "$ARMS")
    while IFS='|' read -r label envs args; do
      [ -n "${label:-}" ] || continue
      arm "$label" "${envs:-}" "${args:-}" "$pass"
    done <<< "$ordered"
  done
  summarize
else
  run cpu
  run npu
fi
