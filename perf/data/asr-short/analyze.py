# Per-utterance marginal cost from the campaign: (10-file process - 1-file process) / 9, per pass.
import sys, os, re, glob, statistics as st, collections
R, C = sys.argv[1], sys.argv[2]
ARMS = ['cpu', 'cpup', 'def', 'forced', 'forcedp']
def rtime(tag):
    e, u, s = open(f'{R}/{tag}.time').read().split()[:3]; return float(e), float(u) + float(s)
def rperf(tag):
    d = {}
    for l in open(f'{R}/{tag}.perf'):
        f = l.strip().split(',')
        if len(f) > 3 and f[0] not in ('', '<not counted>', '<not supported>'):
            d[f[2]] = float(f[0])
    return d.get('armv8_cortex_a76/cycles/u', 0) + d.get('armv8_cortex_a55/cycles/u', 0)
def norm(t): return re.sub(r"[^a-z' ]", ' ', t.lower().replace('-', ' ')).split()
def wer(ref, hyp):
    r, h = ref, hyp; d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]; d[j] = min(d[j] + 1, d[j-1] + 1, prev + (r[i-1] != h[j-1])); prev = cur
    return d[len(h)]
reps = sorted({int(m.group(1)) for f in glob.glob(f'{R}/*.time') for m in [re.search(r'\.r(\d+)\.time$', f)] if m})
buckets = sorted({os.path.basename(f).split('.')[1] for f in glob.glob(f'{R}/*.n10.r1.time')})
res = collections.defaultdict(list)
for b in buckets:
    for a in ARMS:
        for r in reps:
            t1, t10 = f'{a}.{b}.n1.r{r}', f'{a}.{b}.n10.r{r}'
            if not (os.path.exists(f'{R}/{t10}.time') and os.path.exists(f'{R}/{t1}.time')): continue
            (w1, c1), (w10, c10) = rtime(t1), rtime(t10)
            res[(b, a)].append(((w10 - w1) / 9, (c10 - c1) / 9, (rperf(t10) - rperf(t1)) / 9 / 1e9))
refs = {b: [l.rstrip('\n').split('\t') for l in open(f'{C}/{b}/ref.tsv')] for b in buckets}
def trans(a, b, r=1):
    order = [os.path.basename(l.split('Processing file: ')[1].strip())[:-5] for l in open(f'{R}/{a}.{b}.n10.r{r}.err') if 'Processing file: ' in l]
    lines = [l.rstrip('\n') for l in open(f'{R}/{a}.{b}.n10.r{r}.out')]
    assert len(order) == len(lines) == 10, (a, b, len(order), len(lines))
    d = dict(zip(order, lines)); return [d[u] for u, _, _ in sorted(refs[b])]
print(f"{'bucket':6} {'arm':8} {'n':>2} {'wall/utt s':>11} {'core-s/utt':>11} {'Gcyc/utt':>9} {'wall x':>7} {'freed':>7} {'WER%':>6} {'same txt':>8}")
for b in buckets:
    base = res.get((b, 'cpu'))
    if not base: continue
    bw, bc = st.median(x[0] for x in base), st.median(x[1] for x in base)
    ref_by = sorted(refs[b])
    for a in ARMS:
        v = res.get((b, a))
        if not v: continue
        w, c, g = (st.median(x[i] for x in v) for i in range(3))
        # paired within pass
        pw = [base[i][0] / v[i][0] for i in range(min(len(base), len(v)))]
        pc = [1 - v[i][1] / base[i][1] for i in range(min(len(base), len(v)))]
        try:
            hyp = trans(a, b); errs = sum(wer(norm(t), norm(h)) for (_, _, t), h in zip(ref_by, hyp)); nw = sum(len(norm(t)) for _, _, t in ref_by)
            same = sum(h1 == h2 for h1, h2 in zip(hyp, trans('cpu', b)))
            ws, ss = f'{100*errs/nw:6.1f}', f'{same:>5}/10'
        except Exception as e:
            ws, ss = '   n/a', '     n/a'
        print(f"{b:6} {a:8} {len(v):>2} {w:11.3f} {c:11.3f} {g:9.2f} {st.median(pw):7.3f} {100*st.median(pc):6.1f}% {ws} {ss}   passes wall x {[round(x,3) for x in pw]} freed {[round(100*x,1) for x in pc]}")
