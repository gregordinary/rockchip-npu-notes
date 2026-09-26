#!/usr/bin/env python3
"""pair-ratio-sd.py FILE.md [FILE.md ...] -- per-pass paired ratios and their spread.

bench-llm.sh writes one <!--DATA pass arm test t/s--> line per timed arm. The statistic that
decides whether a knob resolved is the ratio formed WITHIN a pass, arm over base, and the sd of
that ratio across passes. This computes exactly that, and nothing else.

WHY THIS IS NOT THE ARM'S SPREAD, which is the statistic it is easiest to read by mistake. An
arm's spread across passes carries every level shift the board took between them; the paired
ratio divides out whatever the two arms shared inside a pass. A knob can be perfectly resolved
while both arms swing 12%, and a knob can be unresolvable while each arm looks tight. Only the
ratio's own sd bounds what the unit can measure -- se = sd / sqrt(passes), and a knob smaller
than about 2 se does not resolve.

The base arm is the first arm seen in the file unless --base names one.

  --base ARM    denominator arm (default: the first arm in the file)
  --test NAME   restrict to one test row (default: all, reported separately)
"""
import re
import sys
import math

DATA = re.compile(r"<!--DATA\s+(.*?)-->", re.S)


def main(argv):
    base = None
    test_filter = None
    files = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--base":
            i += 1
            base = argv[i]
        elif a == "--test":
            i += 1
            test_filter = argv[i]
        else:
            files.append(a)
        i += 1
    if not files:
        print(__doc__)
        return 2

    # rows[(test, pass)][arm] = t/s ; arm order preserved from first appearance
    rows = {}
    order = []
    for path in files:
        text = open(path, encoding="utf-8").read()
        for m in DATA.finditer(text):
            f = m.group(1).split("\t")
            if len(f) < 4:
                continue
            p, arm, test, ts = f[0].strip(), f[1].strip(), f[2].strip(), f[3].strip()
            if test_filter and test != test_filter:
                continue
            try:
                v = float(ts)
            except ValueError:
                continue
            rows.setdefault((test, p), {})[arm] = v
            if arm not in order:
                order.append(arm)

    if not rows:
        print("no <!--DATA--> rows matched")
        return 1

    b = base or order[0]
    if b not in order:
        print("base arm %r not present; arms are %s" % (b, ", ".join(order)))
        return 1

    tests = sorted({t for (t, _) in rows})
    for test in tests:
        passes = sorted(
            (p for (t, p) in rows if t == test), key=lambda x: (len(x), x)
        )
        print("== %s  base=%s  %d passes ==" % (test, b, len(passes)))
        for arm in order:
            if arm == b:
                continue
            rs, missing = [], 0
            for p in passes:
                d = rows[(test, p)]
                if arm in d and b in d and d[b]:
                    rs.append(d[arm] / d[b])
                else:
                    missing += 1
            if not rs:
                print("  %-10s no paired passes" % arm)
                continue
            n = len(rs)
            mean = sum(rs) / n
            sd = math.sqrt(sum((r - mean) ** 2 for r in rs) / (n - 1)) if n > 1 else 0.0
            se = sd / math.sqrt(n) if n > 1 else 0.0
            print(
                "  %-10s mean %.4f  sd %.4f (%.1f%%)  se %.4f (%.1f%%)  n=%d%s"
                % (
                    arm,
                    mean,
                    sd,
                    100 * sd / mean,
                    se,
                    100 * se / mean,
                    n,
                    "  MISSING %d" % missing if missing else "",
                )
            )
            print("             per-pass " + " ".join("%.4f" % r for r in rs))
            if n > 1:
                lo, hi = mean - 2 * se, mean + 2 * se
                verdict = "resolved" if (lo > 1.0 or hi < 1.0) else "STRADDLES 1.00"
                print("             2se band %.4f-%.4f  %s" % (lo, hi, verdict))
        # arm spreads, printed second and labelled, because it is the statistic read by mistake
        print("  -- arm spreads (NOT the statistic above; a level shift lands here) --")
        for arm in order:
            vs = [rows[(test, p)][arm] for p in passes if arm in rows[(test, p)]]
            if len(vs) > 1:
                m = sum(vs) / len(vs)
                print(
                    "     %-10s %.2f-%.2f  range %.1f%% of mean  n=%d"
                    % (arm, min(vs), max(vs), 100 * (max(vs) - min(vs)) / m, len(vs))
                )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
