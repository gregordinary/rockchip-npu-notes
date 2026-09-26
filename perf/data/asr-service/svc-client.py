#!/usr/bin/env python3
"""Drive a running whisper-server the way OpenWebRX+ does, and bill it.

Posts each chunk as multipart form-data with a single `file` field (application/octet-stream
WAV), one request at a time, and records per request: wall latency, and the server process's
CPU core-seconds (utime+stime of the whole process from /proc/PID/stat, sampled immediately
before and after the request), plus the returned text.

usage: svc-client.py --url URL --pid PID --set NAME --dir CHUNKDIR --out results.tsv [--warm N]
"""
import argparse, json, os, sys, time, urllib.request

def cpu_seconds(pid):
    with open(f"/proc/{pid}/stat") as f:
        s = f.read()
    # fields after the ")" of comm; utime=14 stime=15 (1-based in the full line)
    rest = s[s.rindex(")") + 2:].split()
    utime, stime = int(rest[11]), int(rest[12])
    return (utime + stime) / os.sysconf("SC_CLK_TCK")

def post(url, wav_bytes, fields=()):
    boundary = "----WhisperFormBoundary7MA4YWxkTrZu0gW"
    parts = []
    for kv in fields:   # the same shape OpenWebRX+ would add: one text field per line
        k, v = kv.split("=", 1)
        parts += [f"--{boundary}".encode(), f'Content-Disposition: form-data; name="{k}"\r\n'.encode(), v.encode()]
    parts += [
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="file"; filename="file"',
        b"Content-Type: application/octet-stream\r\n",
        wav_bytes,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    payload = b"\r\n".join(parts)
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="")
    ap.add_argument("--rep", default="0")
    ap.add_argument("--warm", type=int, default=0, help="requests to send and discard first")
    ap.add_argument("--field", action="append", default=[], help="extra multipart form field k=v, sent with every request")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.dir) if f.endswith(".wav"))
    new = not os.path.exists(args.out)
    out = open(args.out, "a")
    if new:
        out.write("arm\trep\tset\tchunk\taudio_s\twall_s\tcpu_s\ttext\n")
    texts = []
    for i, fn in enumerate(files):
        wav = open(os.path.join(args.dir, fn), "rb").read()
        audio_s = (len(wav) - 44) / 2 / 12000
        if i < args.warm:
            post(args.url, wav, args.field)
            continue
        c0 = cpu_seconds(args.pid); t0 = time.monotonic()
        res = post(args.url, wav, args.field)
        t1 = time.monotonic(); c1 = cpu_seconds(args.pid)
        text = res.get("text", "").replace("\n", " ").replace("\t", " ").strip()
        texts.append(text)
        out.write(f"{args.arm}\t{args.rep}\t{args.set}\t{fn}\t{audio_s:.2f}\t{t1-t0:.3f}\t{c1-c0:.3f}\t{text}\n")
        out.flush()
    out.close()
    # the concatenated transcript, for scoring against the stream's reference
    tdir = os.path.join(os.path.dirname(args.out) or ".", "transcripts")
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, f"{args.arm}__{args.rep}__{args.set.replace('/', '_')}.txt"), "w") as f:
        f.write(" ".join(texts) + "\n")

if __name__ == "__main__":
    main()
