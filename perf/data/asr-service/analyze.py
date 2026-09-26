#!/usr/bin/env python3
"""Aggregate the service campaign TSV: per arm and set, CPU core-seconds per audio-second,
wall, realtime factor, and pass-paired ratios against a reference arm.

usage: analyze.py campaign.tsv [--ref npu]
"""
import argparse, csv, statistics as st
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--ref", default="npu")
    ap.add_argument("--drop-outliers", action="store_true", help="drop requests > 2.5x the arm/set median wall (fallback ladders)")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.tsv), delimiter="\t"))
    for r in rows:
        r["cond"] = r["set"].split("/")[1]
        r["clen"] = r["set"].split("/")[2]
        for k in ("audio_s", "wall_s", "cpu_s"):
            r[k] = float(r[k])
    # per (arm, rep, cond): sums
    agg = defaultdict(lambda: dict(audio=0.0, wall=0.0, cpu=0.0, n=0, slow=0))
    med = defaultdict(list)
    for r in rows:
        med[(r["arm"], r["cond"])].append(r["wall_s"])
    for r in rows:
        m = st.median(med[(r["arm"], r["cond"])])
        slow = r["wall_s"] > 2.5 * m
        a = agg[(r["arm"], r["rep"], r["cond"])]
        if slow: a["slow"] += 1
        if args.drop_outliers and slow:
            continue
        a["audio"] += r["audio_s"]; a["wall"] += r["wall_s"]; a["cpu"] += r["cpu_s"]; a["n"] += 1
    print(f"{'arm':16} {'rep':>3} {'cond':6} {'n':>2} {'slow':>4} {'cpu/audio_s':>11} {'core-s/req':>10} {'wall/req':>8} {'rt':>5}")
    for (arm, rep, cond), a in sorted(agg.items()):
        if a["n"] == 0: continue
        print(f"{arm:16} {rep:>3} {cond:6} {a['n']:>2} {a['slow']:>4} {a['cpu']/a['audio']:11.3f} {a['cpu']/a['n']:10.2f} {a['wall']/a['n']:8.2f} {a['audio']/a['wall']:5.1f}")
    # pass-paired ratios vs ref (cpu per audio second), per cond
    print(f"\nratio of CPU core-seconds per audio-second against '{args.ref}', paired within a pass (lower is better)")
    arms = sorted(set(r["arm"] for r in rows))
    conds = sorted(set(r["cond"] for r in rows))
    reps = sorted(set(r["rep"] for r in rows))
    print(f"{'arm':16} " + " ".join(f"{c:>14}" for c in conds) + "   all-conds")
    for arm in arms:
        cells = []; allr = []
        for c in conds:
            rs = []
            for rep in reps:
                a = agg.get((arm, rep, c)); b = agg.get((args.ref, rep, c))
                if a and b and a["n"] and b["n"]:
                    rs.append((a["cpu"]/a["audio"]) / (b["cpu"]/b["audio"]))
            allr += rs
            cells.append(f"{st.mean(rs):6.3f} ({len(rs)})" if rs else f"{'-':>10}")
        tot = f"{st.mean(allr):6.3f} +-{(st.pstdev(allr) if len(allr)>1 else 0):.3f}" if allr else "-"
        print(f"{arm:16} " + " ".join(f"{x:>14}" for x in cells) + f"   {tot}")

if __name__ == "__main__":
    main()
