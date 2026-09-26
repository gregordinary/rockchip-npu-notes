#!/bin/bash
# Runs trackd28 then trackd29 back to back, one shell per driver so no OUTD leaks across.
# Its own marker is CHAIN3-DONE, which neither driver writes.
set -u
CL=${CL:-$PWD/chain3.log}
: > "$CL"
{
  echo "[$(date -Is)] chain3 start"
  for c in 0 4 6; do echo "  cpu$c gov=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor)"; done
  echo "  npu clk=$(sudo -n cat /sys/kernel/debug/clk/clk_summary 2>/dev/null | awk '/scmi_clk_npu/{print $5}')"
  echo "[$(date -Is)] trackd28 begin"
  ( env OUTD=$PWD/trackd28-pinub12bq4 bash "$(dirname "$0")"/trackd28-pin-ub-12bq4.sh )
  echo "[$(date -Is)] trackd28 rc=$? end"
  echo "[$(date -Is)] trackd29 begin"
  ( env OUTD=$PWD/trackd29-busyint9b bash "$(dirname "$0")"/trackd29-busy-intercept-9b.sh )
  echo "[$(date -Is)] trackd29 rc=$? end"
  echo "[$(date -Is)] CHAIN3-DONE"
} >> "$CL" 2>&1
