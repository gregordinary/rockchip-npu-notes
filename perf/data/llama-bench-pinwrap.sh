#!/bin/sh
# Run llama-bench under a CPU affinity mask taken from the environment.
#
# WHY A WRAPPER AND NOT `taskset` IN THE ARM'S ENV FIELD. bench-llm.sh builds its timed command
# as `env $envs ROCKET_LOG_STDERR=1 $BIN ...`, so a `taskset 0xf0` written into the env field is
# expanded BEFORE that assignment and consumes `ROCKET_LOG_STDERR=1` as its command name. The arm
# dies rc=127 in 46 seconds, which reads as a fast arm rather than as a failure: the header, the
# <!--PRED--> line and a full plausible <!--RO--> line are all still written, and only the missing
# <!--DATA--> row says anything is wrong. So the mask travels as a variable and is applied here.
#
# BIN points at this wrapper for EVERY arm, pinned or not, so the wrapper is not itself a
# difference between the arms. `exec` replaces the shell, so the process the readout looks for
# (`pgrep -x llama-bench`) is still named llama-bench.
REAL=${PINWRAP_BIN:?PINWRAP_BIN must name the real llama-bench}
if [ -n "${PIN_MASK:-}" ]; then exec taskset "$PIN_MASK" "$REAL" "$@"; fi
exec "$REAL" "$@"
