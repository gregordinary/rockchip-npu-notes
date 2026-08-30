#!/bin/bash
# bench-llm-dryrun.sh [-- extra bench-llm.sh env]
#
# Dry-run bench-llm.sh against STUBS, off the board, before it is trusted with board time.
# It builds the stubs itself, so they cannot drift from the harness they check.
#
# WHY THIS EXISTS. The arm loop's failure mode is silent: an arm whose env or args were parsed
# wrong still runs, still emits rows, and still summarizes -- under the label it was supposed to
# have. The tab-as-IFS bug had benchmarked the STOCK config under the TUNED label, and nothing in
# the output said so. The only check that catches that class is a binary that ECHOES ITS ARGV AND
# ENVIRONMENT, so what each arm actually received is readable next to the label it ran under.
#
# It also stubs `sudo`. reset_mem does drop_caches + compact_memory, so running the harness
# unstubbed on a workstation to "just check the syntax" perturbs the machine you are typing on --
# and on a board, it is a memory reset you did not intend to be part of any measurement.
#
# WHAT A PASS LOOKS LIKE, and none of it is checked automatically -- read the three blocks:
#   1. STUB-SUDO lines list what reset_mem WOULD have done. Nothing ran.
#   2. STUB-ARGV / STUB-ENV pairs: each arm's args in argv and its env vars in the environment,
#      with the stock arm receiving NEITHER. An arm's setting appearing on the wrong side, or a
#      ROCKET_* variable missing, is the bug this exists to catch.
#   3. The <!--PRED-->/<!--DATA--> block and the summary: one PRED line per (pass, arm) BEFORE
#      that arm's DATA rows, the arm order ROTATED between passes, and per-pass ratios present.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
mkdir -p "$D/bin"
export STUB_LOG="$D/stub.log"; : > "$STUB_LOG"

cat > "$D/bin/sudo" <<'EOF'
#!/bin/bash
# NEVER executes its argv: the point is that drop_caches / compact_memory must not run here.
echo "STUB-SUDO: $*" >> "$STUB_LOG"
case "$*" in *clk_summary*) echo "  scmi_clk_npu  2  2  0  600000000  0  0  50000" ;; esac
exit 0
EOF
cat > "$D/bin/llama-bench" <<'EOF'
#!/bin/bash
echo "STUB-ARGV: $*" >> "$STUB_LOG"
echo "STUB-ENV : GGML_BACKEND_PATH=${GGML_BACKEND_PATH:-<unset>} $(env | grep -E '^ROCKET_' | sort | tr '\n' ' ')" >> "$STUB_LOG"
case "$*" in *"-n 8 -r 1"*) exit 0 ;; esac          # the discarded warm-up, as in the real run
b=100.00; case "$*" in *"-b 2048"*) b=113.00 ;; esac  # a per-arm level, so pairing is exercised
echo "| model | size | params | backend | threads | test | t/s |"
echo "| ----- | ---: | -----: | ------- | ------: | ---: | --: |"
for t in pp512 pp2048 tg64; do
  printf '| stub | 500 MiB | 800 M | RPC | 8 | %s | %.2f ± 0.21 |\n' "$t" "$b"
done
echo "[moe-int8] residency pre-flight: 21000MB RAM budget, 20480MB NPU IOVA across 5 worker fds" >&2
exit 0
EOF
chmod +x "$D/bin/sudo" "$D/bin/llama-bench"
export PATH="$D/bin:$PATH"

SO=/dev/null BIN="$D/bin/llama-bench" OUT="$D/out.md" ERRD="$D/err" \
TESTS="-p 512,2048 -n 64 -r 2" PASSES=${PASSES:-2} \
ARMS=${ARMS:-$'stock||\nub2048||-b 2048 -ub 2048'} \
bash "$HERE/bench-llm.sh" /stub/model.gguf dryrun "$@" >/dev/null 2>&1
rc=$?

echo "=== 1. what sudo was ASKED to do, and did not ==="
grep -o "STUB-SUDO: .*" "$STUB_LOG" | sort | uniq -c
echo; echo "=== 2. argv and environment per TIMED arm ==="
grep -A1 -- "STUB-ARGV.*-o md" "$STUB_LOG"
echo; echo "=== 3. predictors, data and summary ==="
grep -E '^<!--' "$D/out.md"
sed -n '/#### summary/,$p' "$D/out.md"
echo; echo "bench-llm.sh rc=$rc"
