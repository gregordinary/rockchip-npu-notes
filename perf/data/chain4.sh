#!/bin/bash
# Runs trackd30, trackd31 and trackd32 back to back, one shell per driver so no OUTD leaks across.
# Its own marker is CHAIN4-DONE, which no driver writes. Restores the board to idle defaults at
# the end (ondemand, power/control=auto) so the next session does not inherit a pinned board.
set -u
CL=${CL:-$PWD/chain4.log}
: > "$CL"
{
  echo "[$(date -Is)] chain4 start"
  for c in 0 4 6; do echo "  cpu$c gov=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor)"; done
  echo "  npu clk=$(sudo -n cat /sys/kernel/debug/clk/clk_summary 2>/dev/null | awk '/scmi_clk_npu/{print $5}')"
  for d in 30-fa-timing-12b:trackd30-fatiming12b 31-qres512-9b:trackd31-qres512-9b 32-fa-pin-2x2-12b:trackd32-fa-pin-12b; do
    s=${d%%:*}; o=${d##*:}
    echo "[$(date -Is)] trackd$s begin"
    ( env OUTD=$PWD/$o bash "$(dirname "$0")"/trackd$s.sh )
    echo "[$(date -Is)] trackd$s rc=$? end"
  done
  for c in 0 4 6; do echo ondemand | sudo -n tee /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor >/dev/null; done
  for d in /sys/bus/platform/drivers/rocket/*.npu; do echo auto | sudo -n tee $d/power/control >/dev/null; done
  echo "[$(date -Is)] board restored: ondemand, power/control=auto"
  echo "[$(date -Is)] CHAIN4-DONE"
} >> "$CL" 2>&1
