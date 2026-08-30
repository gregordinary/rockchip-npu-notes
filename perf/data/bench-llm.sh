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
# happens to match a default. The label `cpu` is special-cased to drop GGML_BACKEND_PATH, so the
# absolute reference can ride along in the same pass at the same clock.
#
# Every arm runs under ROCKET_LOG_STDERR=1 and keeps its stderr, because the mode/budget/OUTCOME
# lines are what make a row self-documenting.
#
# PASSIVE PREDICTORS. Every timed arm also emits a <!--PRED pass arm k=v ...--> line, read once at
# the start of the timed run: MemAvailable, AnonHugePages/Hugepagesize, and the per-order free-page
# vector from /proc/buddyinfo. They cost two file reads and no process, and they are there so that
# the per-process spread can be attacked from rows already taken rather than from a campaign run
# later. PRED_SAMPLE=1 adds a per-thread last-CPU histogram, which is NOT free -- see pred_sample. The outcome half is load-bearing: a residency arm
# that silently placed nothing produces the same rows and the same t/s as one that placed
# everything and gained nothing, and only the teardown split tells them apart.
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
EXTRA="$*"                                    # e.g. "-b 2048 -ub 2048" for quant streaming
# TESTS is overridable because the flag axis multiplies the cost by the number of arms: the
# headline "what does tuning buy" table needs only the pp2048 row, and paying for the decode and
# interactive rows on every arm is what turns a model into an hour.
TESTS=${TESTS:-"-p 512,1024,2048 -n 64 -pg 2048,128 -r 2"}

run() { # $1 = npu|cpu
  local mode="$1" envs=""
  [ "$mode" = npu ] && envs="GGML_BACKEND_PATH=$SO ROCKET_KACC=1"
  reset_mem
  # discarded warmup: spin the NPU clock off idle before the measured run
  env $envs $BIN -m "$MODEL" -p 512 -n 8 -r 1 $EXTRA >/dev/null 2>&1
  echo "### $LABEL  [$mode]  $(date +%T)  clk=$(sudo cat /sys/kernel/debug/clk/clk_summary 2>/dev/null | awk '/scmi_clk_npu/{printf "%d MHz",$5/1e6; exit}')" | tee -a "$OUT"
  env $envs $BIN -m "$MODEL" $TESTS $EXTRA -o md 2>/dev/null | tee -a "$OUT"
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
  awk -v p="$1" -v l="$2" '
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

arm() { # $1 = label, $2 = env assignments, $3 = extra llama-bench args, $4 = pass number
  local label="$1" envs="$2" args="$3" pass="${4:-1}" tag
  tag="$LABEL.$label.p$pass"
  [ "$label" = cpu ] || envs="GGML_BACKEND_PATH=$SO $envs"
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
  env $envs ROCKET_LOG_STDERR=1 $BIN -m "$MODEL" $TESTS $args -o md 2>"$ERRD/$tag.err" | tee -a "$OUT" | tee "$ERRD/$tag.md" >/dev/null
  local rc=${PIPESTATUS[0]}
  if [ -n "$spid" ]; then
    wait "$spid" 2>/dev/null
    [ -s "$ERRD/$tag.cpu" ] && printf '<!--PRED %s\t%s\tlastcpu=%s-->\n' \
        "$pass" "$label" "$(cat "$ERRD/$tag.cpu")" | tee -a "$OUT"
  fi
  [ "$rc" -eq 0 ] || echo "    ARM FAILED rc=$rc -- see $ERRD/$tag.err" | tee -a "$OUT"
  # One machine-readable datum per (pass, arm, test), so the summary below is computed from the
  # numbers rather than re-parsed out of prose. Kept as an HTML comment: the file is markdown.
  awk -F'|' -v p="$pass" -v l="$label" 'NF>6 && $(NF-2) ~ /pp|tg/ {
        t=$(NF-2); v=$(NF-1); gsub(/ /,"",t); gsub(/ /,"",v); sub(/±.*/,"",v);
        if (v+0 > 0) printf "<!--DATA %s\t%s\t%s\t%s-->\n", p, l, t, v }' \
      "$ERRD/$tag.md" | tee -a "$OUT"
  grep -hE "residency pre-flight|resident on the NPU|streamed via|admission first declined|budget reached|experts exercised|resident budget|MM_ASYM|K-accum" \
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

echo "== $LABEL  $(date) ==" | tee -a "$OUT"
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
