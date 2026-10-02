# Neighbor (run2), LLM crossover (run4), sherpa (run5) readouts.
import sys, os, re, json, glob, statistics as st, collections
base = sys.argv[1]
def jl(p): return [json.loads(l) for l in open(p) if l.strip().startswith('{')]
r2 = f'{base}/run2'
if os.path.isdir(r2):
    print('== neighbor: llama-bench tg64 t/s (Llama-3.2-3B Q4_K_M, 4 A76 threads) beside a b02 STT stream')
    tg = collections.defaultdict(dict); stt = collections.defaultdict(dict)
    for f in glob.glob(f'{r2}/*.nb.jsonl'):
        a, r = os.path.basename(f).split('.')[:2]
        if r == 'rw0': continue
        rows = [x for x in jl(f) if x.get('n_gen', 0) > 0]
        if rows: tg[a][r] = rows[0]['avg_ts']
        e = f'{r2}/{a}.{r}.stt.err'
        if os.path.exists(e):
            enc = [float(m.group(1)) for m in re.finditer(r'encode time =\s+([0-9.]+) ms /\s+(\d+) runs', open(e).read())]
            runs = [int(m.group(2)) for m in re.finditer(r'encode time =\s+([0-9.]+) ms /\s+(\d+) runs', open(e).read())]
            tot = [float(m.group(1)) for m in re.finditer(r'total time =\s+([0-9.]+) ms', open(e).read())]
            if len(tot) > 2:
                per = [(tot[i+1] - tot[i]) for i in range(len(tot) - 1)]
                stt[a][r] = st.median(per)
    for a in ['alone', 'cpu', 'forced', 'forcedp']:
        v = tg.get(a, {})
        rat = [v[r] / tg['alone'][r] for r in v if r in tg.get('alone', {})]
        print(f"{a:8} tg {[round(v[r],2) for r in sorted(v)]} median {st.median(v.values()) if v else 'n/a':} vs alone {[round(x,3) for x in rat]}  stt ms/utt under contention {[round(stt[a][r]) for r in sorted(stt.get(a,{}))]}")
    for r in sorted(tg.get('cpu', {})):
        if r in tg.get('forced', {}): print(f"  pass {r}: forced/cpu {tg['forced'][r]/tg['cpu'][r]:.3f}  forcedp/cpu {tg.get('forcedp',{}).get(r,0)/tg['cpu'][r]:.3f}")
r4 = f'{base}/run4'
if os.path.isdir(r4):
    print('== LLM crossover, Llama-3.2-3B F16 pp t/s, NPU (floor 4) / CPU')
    d = collections.defaultdict(dict)
    for f in glob.glob(f'{r4}/*.jsonl'):
        a, r = os.path.basename(f).split('.')[:2]
        for x in jl(f):
            if x.get('n_prompt', 0) > 0: d[(a, x['n_prompt'])][r] = x['avg_ts']
    for p in (16, 32, 64, 128):
        c, n = d.get(('cpu', p), {}), d.get(('npu', p), {})
        rat = [n[r] / c[r] for r in n if r in c]
        print(f"pp{p:<4} cpu {[round(c[r],1) for r in sorted(c)]} npu {[round(n[r],1) for r in sorted(n)]} ratio {[round(x,3) for x in rat]}")
r5 = f'{base}/run5'
if os.path.isdir(r5):
    print('== sherpa-onnx Parakeet TDT int8, 4 A76 threads, per-utterance medians over passes 1-2 (pass 0 dropped)')
    for m in ('v2', 'v3'):
        p = f'{r5}/{m}.jsonl'
        if not os.path.exists(p): continue
        rows = jl(p); hdr = rows[0]; rows = [x for x in rows[1:] if x.get('pass', 0) > 0]
        by = collections.defaultdict(list)
        for x in rows: by[x['bucket']].append(x)
        print(m, 'load', round(hdr['load_s'], 2), 's')
        for b in sorted(by):
            print(f"  {b} wall {1000*st.median(x['wall'] for x in by[b]):6.0f} ms  cpu {st.median(x['cpu'] for x in by[b]):5.2f} core-s")
