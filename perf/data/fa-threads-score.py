#!/usr/bin/env python3
"""Score the ROCKET_FA_THREADS A/B from a trackd33 output directory.

WHAT IT READS, AND WHY NOT THE WALL. The deliverable is `G_k`, the handler's gather plus
scatter seconds per 2048-token prefill at `ROCKET_FA_THREADS=k`. That interval is what the
prediction's band is written in, because the handler runs on the ggml dispatch thread while
the scheduler waits: the bracket IS wall and no exchange rate enters. The wall ratio is the
secondary reading and is reported beside it, never in place of it.

Per-prefill is the unit because the `ROCKET FA total` line accumulates over every offloaded op
in the process, and one llama-bench invocation at `-r 2` runs three prefills. The prefill count
is DERIVED from the op count rather than assumed: ops per prefill is read from the arms
themselves, so a run whose micro-batch count or layer count differs is caught instead of being
silently divided by three.

PAIRED WITHIN A PASS. Every ratio pairs the two arms of the SAME pass, then averages the
ratios. This board drifts a level between passes that is larger than the effect, and a ratio of
means does not remove it. The per-pass ratios are printed beside the mean because a set that
straddles 1.00 is a result about the instrument and not about the knob.
"""
import re, sys, os, glob, statistics

FA_RE = re.compile(
    r"ROCKET FA total\(ms\):\s*gather=(\d+)\s*\([^)]*\)\s*compute=(\d+)\s*\([^)]*\)\s*"
    r"scatter=(\d+)\s*\([^)]*\)\s*over\s+(\d+)\s+ops,\s*n_kv\s*\[(\d+)\.\.(\d+)\]")
DATA_RE = re.compile(r"<!--DATA\s+(\S+)\t(\S+)\t(\S+)\t([0-9.]+)-->")
ERR_RE  = re.compile(r"\.([^.]+)\.p(\d+)\.err$")

def main(outd):
    md = glob.glob(os.path.join(outd, "*.md"))
    if not md:
        sys.exit(f"no .md in {outd}")
    text = open(md[0], errors="replace").read()

    # wall: (pass, arm) -> t/s
    wall = {}
    for p, arm, test, ts in DATA_RE.findall(text):
        wall[(int(p), arm)] = float(ts)

    # interval: (pass, arm) -> (gather_ms, compute_ms, scatter_ms, ops, kv_lo, kv_hi)
    iv = {}
    for f in sorted(glob.glob(os.path.join(outd, "err", "*.err"))):
        m = ERR_RE.search(f)
        if not m:
            continue
        arm, p = m.group(1), int(m.group(2))
        hits = FA_RE.findall(open(f, errors="replace").read())
        if hits:
            g, c, s, ops, lo, hi = hits[-1]
            iv[(p, arm)] = (float(g), float(c), float(s), int(ops), int(lo), int(hi))

    arms = sorted({a for _, a in iv} | {a for _, a in wall})
    passes = sorted({p for p, _ in iv} | {p for p, _ in wall})
    if not arms:
        sys.exit("no arms found: no FA total line in any .err")

    # The op count is a control, not an input: every arm must run the same graph, or a ratio
    # between two arms is not a ratio of the same thing.
    opset = {v[3] for v in iv.values()}
    kvset = {(v[4], v[5]) for v in iv.values()}
    print(f"# trackd33 FA_THREADS, {len(arms)} arms x {len(passes)} passes")
    print(f"  arms          : {', '.join(arms)}")
    print(f"  FA op counts  : {sorted(opset)}   {'OK (one graph)' if len(opset)==1 else 'DIFFER -- ratios are not comparable'}")
    print(f"  n_kv ranges   : {sorted(kvset)}   {'OK' if len(kvset)==1 else 'DIFFER'}")
    if len(opset) != 1:
        print("  REFUSING to score: the arms did not run the same graph.")
        return 1
    ops = opset.pop()
    # The divisor comes from the RUN, not from the model. llama-bench does one warm-up prefill
    # plus `-r` timed ones, so the process runs r+1 of them; ops-per-prefill is then derived and
    # reported rather than carried as a per-model constant that silently goes wrong on the next
    # model or micro-batch size. gemma4-12b at -ub 512 offloads 3 of 4 micro-batches per layer
    # over 48 layers, so it should read 144.
    m = re.search(r"-r\s+(\d+)", text)
    if m:
        prefills = int(m.group(1)) + 1
        src = f"-r {m.group(1)} in the header, so {prefills} prefills"
    else:
        prefills = 3
        src = "no -r found in the header; assuming the 3 prefills of -r 2"
    if ops % prefills:
        print(f"  WARNING: {ops} ops does not divide by {prefills} prefills; "
              f"an arm did not run the graph the header describes.")
    print(f"  prefills/proc : {prefills}  ({src})")
    print(f"  ops/prefill   : {ops / prefills:g}  (derived; gemma4-12b at -ub 512 reads 144)")
    print()

    print("## Interval per prefill (seconds), from the ROCKET FA total line")
    print("| arm | pass | gather | scatter | G = g+s | compute | t/s |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    G = {}
    for a in arms:
        for p in passes:
            if (p, a) not in iv:
                continue
            g, c, s, _, _, _ = iv[(p, a)]
            gs, ss, cs = g/1e3/prefills, s/1e3/prefills, c/1e3/prefills
            G[(p, a)] = gs + ss
            ts = wall.get((p, a))
            print(f"| {a} | {p} | {gs:.2f} | {ss:.2f} | {gs+ss:.2f} | {cs:.2f} | "
                  f"{ts if ts is None else format(ts, '.2f')} |")
    print()

    base = arms[0]
    print(f"## Ratios, paired within a pass against [{base}]")
    # Orientation, in symbols and once: r = G_k/G_1, so BELOW 1 is the gain. The wall column is
    # t_1/t_k, wall SECONDS, so ABOVE 1 is the gain -- and since llama-bench reports throughput,
    # t_1/t_k is (t/s at k) / (t/s at 1), which is what is computed. The two columns therefore
    # move in opposite directions on a win, and that is not a sign error.
    print("| arm | r = G_k/G_1 (mean) | per-pass | wall t_1/t_k (mean) | per-pass |")
    print("|---|---:|---|---:|---|")
    out = {}
    for a in arms:
        rs = [G[(p, a)] / G[(p, base)] for p in passes
              if (p, a) in G and (p, base) in G and G[(p, base)]]
        ws = [wall[(p, a)] / wall[(p, base)] for p in passes
              if (p, a) in wall and (p, base) in wall and wall[(p, base)]]
        if a == base:
            print(f"| {a} | -- | -- | -- | -- |")
            continue
        rm = statistics.mean(rs) if rs else float("nan")
        wm = statistics.mean(ws) if ws else float("nan")
        out[a] = (rm, wm, rs, ws)
        print(f"| {a} | {rm:.3f} | {' '.join(f'{x:.3f}' for x in rs)} | "
              f"{wm:.4f} | {' '.join(f'{x:.4f}' for x in ws)} |")
    print()

    # The prediction's axis is tiled, so every landing place is registered in advance.
    if "fa_t4" in out:
        r, w, rs, _ = out["fa_t4"]
        g1 = statistics.mean([G[(p, base)] for p in passes if (p, base) in G])
        print("## Prediction 2026-09-07, scored")
        print(f"  G_1 measured here : {g1:.2f} s per prefill (the prediction's input was 5.44 s)")
        print(f"  r = G_4/G_1       : {r:.3f}   (per pass: {' '.join(f'{x:.3f}' for x in rs)})")
        print(f"  wall t_1/t_4      : {w:.4f}   (band 1.032-1.041)")
        if r < 0.29:
            v = ("BELOW THE BAND. The walks were compute-bound rather than bandwidth-bound, "
                 "and the memory-traffic argument that set the band's floor is wrong.")
        elif r <= 0.44:
            v = "IN THE BAND (0.29-0.44). HIT."
        elif r <= 0.63:
            v = ("0.44-0.63: the walks ARE the interval's bottleneck, and bandwidth caps the "
                 "gain harder than the 2.4 s estimate. MISS, band too optimistic.")
        else:
            v = ("ABOVE 0.63: the loops were never the bottleneck of their own interval. This "
                 "retires the 5.6% cap as a cap on THIS lever. MISS.")
        print(f"  verdict           : {v}")
    if "fa_t2" in out:
        r2 = out["fa_t2"][0]
        r4 = out["fa_t4"][0] if "fa_t4" in out else float("nan")
        print()
        print("## Does it saturate? (the question the third arm buys)")
        print(f"  G_2/G_1 = {r2:.3f}, G_4/G_1 = {r4:.3f}; ideal core scaling is 0.500 and 0.250.")
        print("  Two workers close to 0.5 with four close to 0.25 is core-limited; the two "
              "readings close together is bandwidth-limited.")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
