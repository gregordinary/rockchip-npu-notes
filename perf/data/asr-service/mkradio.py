#!/usr/bin/env python3
"""Build OpenWebRX+-shaped whisper-server request sets from LibriSpeech test-clean.

The client (owrx/transcribe.py) accumulates squelch-gated 12 kHz PCM and posts whatever
arrived in the last chunkSeconds of wall time, cutting wherever the timer falls. So the
shape to reproduce is: a continuous speech stream (silence already removed by the squelch),
cut at fixed wall-clock lengths, delivered as 12 kHz 16-bit mono WAV.

Per stream: one LibriSpeech chapter's utterances in order, 0.4 s gaps, ~STREAM_S seconds.
Per channel condition:
  clean  : the stream as is, resampled to 12 kHz
  ssb15  : SSB-like voice channel, band-pass 300-2700 Hz, additive noise at 15 dB SNR,
           soft limiting, +25 Hz carrier offset, slow fading (+-3 dB, 0.3 Hz)
  ssb5   : the same at 5 dB SNR, +40 Hz offset, deeper fading (+-6 dB)
Per chunking: 20 s and 30 s fixed cuts.

Writes: OUT/<stream>/<cond>/c<len>/NNN.wav, OUT/<stream>/ref.txt, OUT/manifest.json
"""
import argparse, json, os, subprocess, sys, wave
import numpy as np
from scipy import signal

SR = 16000
OUT_SR = 12000

def load_flac(path):
    """Decode with ffmpeg to 16 kHz mono float32."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()

def find_chapters(root):
    """Yield (speaker, chapter, [(utt_id, flac_path, text)...]) for every chapter."""
    for spk in sorted(os.listdir(root), key=int):
        sd = os.path.join(root, spk)
        if not os.path.isdir(sd):
            continue
        for ch in sorted(os.listdir(sd), key=int):
            cd = os.path.join(sd, ch)
            trans = os.path.join(cd, f"{spk}-{ch}.trans.txt")
            if not os.path.exists(trans):
                continue
            utts = []
            for line in open(trans):
                uid, text = line.strip().split(" ", 1)
                utts.append((uid, os.path.join(cd, uid + ".flac"), text))
            yield spk, ch, utts

def build_stream(utts, stream_s, gap_s=0.4):
    """Concatenate utterances in order until stream_s; return (audio, ref_text, n_utts)."""
    parts, texts, total = [], [], 0.0
    gap = np.zeros(int(gap_s * SR), dtype=np.float32)
    for uid, path, text in utts:
        a = load_flac(path)
        parts.append(a); parts.append(gap); texts.append(text)
        total += len(a) / SR + gap_s
        if total >= stream_s:
            break
    audio = np.concatenate(parts)[: int(stream_s * SR)]
    return audio, " ".join(texts), len(texts)

def ssb_channel(x, snr_db, f_off, fade_db, seed):
    """SSB-like voice channel at 16 kHz."""
    rng = np.random.default_rng(seed)
    bp = signal.firwin(255, [300, 2700], pass_zero=False, fs=SR)
    y = signal.lfilter(bp, [1.0], x)
    # carrier offset via the analytic signal
    t = np.arange(len(y)) / SR
    ya = signal.hilbert(y)
    y = np.real(ya * np.exp(2j * np.pi * f_off * t)).astype(np.float32)
    # slow fading
    fade = 10 ** (fade_db * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 6.28)) / 20)
    y = y * fade
    # noise at the target SNR over the speech-active RMS, band-limited like the receiver
    frame = int(0.02 * SR)
    nfr = len(y) // frame
    rms_f = np.sqrt(np.mean(y[: nfr * frame].reshape(nfr, frame) ** 2, axis=1))
    active = rms_f[rms_f > np.percentile(rms_f, 30)]
    s_rms = np.sqrt(np.mean(active ** 2))
    noise = signal.lfilter(bp, [1.0], rng.standard_normal(len(y)).astype(np.float32))
    noise *= s_rms / np.sqrt(np.mean(noise ** 2)) / (10 ** (snr_db / 20))
    y = y + noise
    # receiver AGC + soft limiting
    y = np.tanh(2.5 * y / (np.max(np.abs(y)) + 1e-9)) * 0.8
    return y.astype(np.float32)

def to_12k_int16(x):
    y = signal.resample_poly(x, 3, 4)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
    return (y * 32767).astype(np.int16)

def write_wav(path, pcm):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = wave.open(path, "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(OUT_SR)
    w.writeframes(pcm.tobytes()); w.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--streams", type=int, default=2)
    ap.add_argument("--stream-seconds", type=int, default=240)
    ap.add_argument("--chunks", default="20,30")
    args = ap.parse_args()

    conds = {
        "clean": None,
        "ssb15": dict(snr_db=15, f_off=25, fade_db=3),
        "ssb5":  dict(snr_db=5,  f_off=40, fade_db=6),
    }
    manifest = {"sr": OUT_SR, "streams": {}}
    n = 0
    for spk, ch, utts in find_chapters(args.root):
        dur = sum(os.path.getsize(p) for _, p, _ in utts)  # rough size proxy
        if dur < 2.5e6:   # skip short chapters (flac ~1 MB/min)
            continue
        audio, ref, nutt = build_stream(utts, args.stream_seconds)
        if len(audio) < args.stream_seconds * SR - SR:
            continue
        name = f"s{spk}-{ch}"
        sdir = os.path.join(args.out, name)
        os.makedirs(sdir, exist_ok=True)
        open(os.path.join(sdir, "ref.txt"), "w").write(ref + "\n")
        entry = {"speaker": spk, "chapter": ch, "utts": nutt, "seconds": len(audio) / SR, "sets": {}}
        for cname, cp in conds.items():
            y = audio if cp is None else ssb_channel(audio, seed=n * 7 + 1, **cp)
            pcm = to_12k_int16(y)
            for clen in (int(c) for c in args.chunks.split(",")):
                step = clen * OUT_SR
                files = []
                for i, s in enumerate(range(0, len(pcm), step)):
                    piece = pcm[s : s + step]
                    if len(piece) < OUT_SR:   # drop a sub-second tail
                        continue
                    p = os.path.join(sdir, cname, f"c{clen}", f"{i:03d}.wav")
                    write_wav(p, piece); files.append(p)
                entry["sets"][f"{cname}/c{clen}"] = files
        manifest["streams"][name] = entry
        n += 1
        print(f"{name}: {nutt} utterances, {len(audio)/SR:.1f} s", file=sys.stderr)
        if n >= args.streams:
            break
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
