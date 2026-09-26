#!/usr/bin/env python3
"""ro-join.py FILE.md [FILE.md ...] -- join a bench run's annotation lines to its t/s rows.

bench-llm.sh writes three machine-readable comments per timed arm -- <!--DATA--> (the t/s),
<!--PRED--> (the board snapshot) and <!--RO--> (the per-process readout) -- all keyed by
(pass, arm). This joins them and reports, per group, whether any covariate tracks the wall.

WHY IT REPORTS THE COVARIATE'S OWN RANGE FIRST. The failure that closed the previous round of
per-process-spread suspects was not a weak correlation, it was NO VARIANCE TO CORRELATE WITH.
MemAvailable and the buddy tail sat flat to under 1.2% while the t/s swung 8-13%, and a
correlation coefficient computed over a flat column is noise dressed as a result. So every
covariate is printed with its own spread beside its correlation, and one whose range is under
`--flat` (default 2%) is marked FLAT and its r is not to be quoted. A null from a flat column
is a statement about the board, not about the covariate.

Correlation is Spearman (rank), because none of these is expected to be linear in t/s and one
outlying process would dominate a Pearson r over five points.

**RHO IS DEGENERATE AT SMALL n AND THE OUTPUT SAYS SO.** Over three points a monotone column
can only score +-1.0 or +-0.5, so |rho| = 1 carries almost no information and a three-pass
matrix unit will show a screenful of perfect correlations. Groups under `--rho-n` (default 5)
are printed with their ranges but their rho is labelled `n too small`. The spread work this
tool exists for needs six or more processes per arm, which is why the campaigns that hunt it
run six passes rather than the matrix default of three.

Groups are (file, arm, test): the knob and the shape are held constant inside one, so what is
left varying between its rows is the per-process lottery -- which is the thing under test.

  --flat PCT     range below this (percent of the median) marks a covariate FLAT (default 2)
  --min-n N      skip groups with fewer than N joined rows (default 3)
  --rho-n N      below this many rows, rho is reported as uninformative (default 5)
  --keys REGEX   restrict the covariate columns to those matching
  --rows         also dump the joined rows as TSV
"""
import math
import re
import sys

DATA = re.compile(r"<!--DATA\s+(.*?)-->", re.S)
PRED = re.compile(r"<!--PRED\s+(.*?)-->", re.S)
RO = re.compile(r"<!--RO\s+(.*?)-->", re.S)
NUM = re.compile(r"^-?\d+(\.\d+)?$")


def kvs(fields):
    out = {}
    for f in fields:
        if "=" not in f:
            continue
        k, _, v = f.partition("=")
        k = k.strip()
        v = v.strip()
        if NUM.match(v):
            out[k] = float(v)
        elif k in ("buddy", "busy"):
            # Vector columns: expand to one covariate per position so a later analysis can
            # pick its own order/CPU rather than inheriting a cut chosen here.
            for i, part in enumerate(v.split(",")):
                if NUM.match(part.strip()):
                    out["%s%d" % (k, i)] = float(part)
    return out


def parse(path):
    text = open(path, errors="replace").read()
    data, ann = {}, {}
    for m in DATA.finditer(text):
        f = m.group(1).split("\t")
        if len(f) >= 4:
            data[(f[0].strip(), f[1].strip(), f[2].strip())] = float(f[3])
    for rx in (PRED, RO):
        for m in rx.finditer(text):
            f = [x for x in m.group(1).split("\t") if x != ""]
            if len(f) < 3:
                continue
            ann.setdefault((f[0].strip(), f[1].strip()), {}).update(kvs(f[2:]))
    rows = []
    for (p, a, t), ts in sorted(data.items()):
        rows.append({"pass": p, "arm": a, "test": t, "ts": ts, **ann.get((p, a), {})})
    return rows


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else 0.0


def spread(xs):
    """Range as a percentage of the median -- the same denominator the matrix quotes."""
    med = sorted(xs)[len(xs) // 2]
    return 100.0 * (max(xs) - min(xs)) / abs(med) if med else 0.0


def main():
    args = sys.argv[1:]
    flat, min_n, rho_n, keyre, dump = 2.0, 3, 5, None, False
    files = []
    i = 0
    while i < len(args):
        if args[i] == "--flat":
            flat = float(args[i + 1]); i += 2
        elif args[i] == "--min-n":
            min_n = int(args[i + 1]); i += 2
        elif args[i] == "--rho-n":
            rho_n = int(args[i + 1]); i += 2
        elif args[i] == "--keys":
            keyre = re.compile(args[i + 1]); i += 2
        elif args[i] == "--rows":
            dump = True; i += 1
        else:
            files.append(args[i]); i += 1
    if not files:
        print(__doc__); sys.exit(2)

    for path in files:
        rows = parse(path)
        if not rows:
            print("## %s -- no <!--DATA--> rows" % path)
            continue
        groups = {}
        for r in rows:
            groups.setdefault((r["arm"], r["test"]), []).append(r)
        print("## %s" % path)
        if dump:
            keys = sorted({k for r in rows for k in r})
            print("\t".join(keys))
            for r in rows:
                print("\t".join(str(r.get(k, "")) for k in keys))
        for (arm, test), rs in sorted(groups.items()):
            if len(rs) < min_n:
                continue
            ts = [r["ts"] for r in rs]
            weak = len(rs) < rho_n
            print("### arm=%s test=%s  n=%d  t/s %.2f-%.2f  spread %.1f%%%s"
                  % (arm, test, len(rs), min(ts), max(ts), spread(ts),
                     "   [n < %d: rho uninformative]" % rho_n if weak else ""))
            cand = sorted({k for r in rs for k in r} - {"pass", "arm", "test", "ts"})
            if keyre:
                cand = [k for k in cand if keyre.search(k)]
            out = []
            for k in cand:
                xs = [r.get(k) for r in rs]
                if any(x is None for x in xs) or len(set(xs)) < 2:
                    continue
                sp = spread(xs)
                out.append((abs(spearman(ts, xs)), k, sp, spearman(ts, xs)))
            if not out:
                print("    (no covariate varies in this group)")
                continue
            out.sort(reverse=True)
            for _, k, sp, r in out[:12]:
                tag = "  FLAT -- r not quotable" if sp < flat else ""
                if weak:
                    tag = "  n too small" + tag
                print("    %-22s range %7.2f%%   rho %+0.3f%s" % (k, sp, r, tag))


if __name__ == "__main__":
    main()
