#!/usr/bin/env python3
"""Word error rate of the service transcripts against each stream's reference.

usage: wer.py --sets SETS_DIR --transcripts DIR [--tsv results.tsv]
Transcript files are named ARM__REP__STREAM_COND_CLEN.txt (svc-client.py writes them).
Normalisation: lowercase, punctuation stripped, small integers spelled out (LibriSpeech
references spell numbers), hyphens split. Numbers of a form the spelling does not cover are
left as digits and count against both arms equally.
"""
import argparse, json, os, re, sys
from collections import defaultdict

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()

def num_words(n):
    if n < 20: return ONES[n]
    if n < 100: return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000: return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + num_words(n % 100))
    if n < 10000: return ONES[n // 1000] + " thousand" + ("" if n % 1000 == 0 else " " + num_words(n % 1000))
    return str(n)

def normalize(text):
    t = text.lower()
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", t)          # [skipping ...], (music)
    t = t.replace("-", " ").replace("'", "")
    t = re.sub(r"\b(\d{1,4})\b", lambda m: num_words(int(m.group(1))), t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return t.split()

def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]

def align_start(ref, hyp, n=12, max_d=5):
    """The first set of an arm drops its warm-up chunks from the transcript, so the hypothesis can
    start tens of seconds into the reference. Find where the hypothesis's first n words sit in the
    reference and score from there; leave the reference whole when no window matches within max_d."""
    if len(hyp) < n or len(ref) <= n:
        return ref
    head = hyp[:n]
    best_i, best_d = 0, edit_distance(head, ref[:n])
    for i in range(1, len(ref) - n):
        d = edit_distance(head, ref[i:i + n])
        if d < best_d:
            best_i, best_d = i, d
            if d == 0:
                break
    return ref[best_i:] if best_d <= max_d else ref

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    refs = {}
    for s in os.listdir(args.sets):
        p = os.path.join(args.sets, s, "ref.txt")
        if os.path.exists(p):
            refs[s] = normalize(open(p).read())
    rows = []
    for fn in sorted(os.listdir(args.transcripts)):
        if not fn.endswith(".txt"):
            continue
        arm, rep, setname = fn[:-4].split("__")
        stream = next((s for s in refs if setname.startswith(s + "_")), None)
        if stream is None:
            continue
        cond = setname[len(stream) + 1:]
        hyp = normalize(open(os.path.join(args.transcripts, fn)).read())
        ref = align_start(refs[stream], hyp)
        d = edit_distance(ref, hyp)
        rows.append(dict(arm=arm, rep=rep, stream=stream, cond=cond, ref_words=len(ref), hyp_words=len(hyp), edits=d, wer=d / len(ref)))
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["stream"], r["cond"])].append(r["wer"])
    print(f"{'arm':22} {'stream':13} {'cond':12} {'n':>2} {'WER%':>6}  per-rep")
    for (arm, stream, cond), ws in sorted(by.items()):
        print(f"{arm:22} {stream:13} {cond:12} {len(ws):>2} {100*sum(ws)/len(ws):6.2f}  " + " ".join(f"{100*w:.1f}" for w in ws))
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)

if __name__ == "__main__":
    main()
