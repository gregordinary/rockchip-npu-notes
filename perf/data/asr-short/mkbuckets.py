# Select LibriSpeech test-clean utterances into duration buckets, deterministic: b02-b12 one per
# speaker in utterance-id order, b01 the ten shortest. Usage: python3 mkbuckets.py OUTDIR
import os, struct, glob, shutil, sys
ROOT = os.path.expanduser('~/asr-data/LibriSpeech/test-clean')
BUCKETS = [('b02', 1.4, 2.6), ('b03', 2.6, 4.0), ('b05', 4.0, 6.0), ('b08', 7.0, 9.0), ('b12', 11.0, 14.0)]
N = 10
def flac_dur(p):
    with open(p, 'rb') as f:
        assert f.read(4) == b'fLaC'
        hdr = f.read(4); si = f.read(34)
    sr = (si[10] << 12) | (si[11] << 4) | (si[12] >> 4)
    tot = ((si[13] & 0x0F) << 32) | struct.unpack('>I', si[14:18])[0]
    return tot / sr, sr
trans = {}
for t in glob.glob(f'{ROOT}/*/*/*.trans.txt'):
    for line in open(t):
        uid, txt = line.rstrip('\n').split(' ', 1); trans[uid] = txt
utts = []
for uid in sorted(trans):
    spk, ch, _ = uid.split('-')
    d, sr = flac_dur(f'{ROOT}/{spk}/{ch}/{uid}.flac'); assert sr == 16000
    utts.append((uid, spk, d))
out = sys.argv[1]
for name, lo, hi in BUCKETS:
    seen, picked = set(), []
    for uid, spk, d in utts:
        if lo <= d < hi and spk not in seen:
            seen.add(spk); picked.append((uid, d))
        if len(picked) == N: break
    os.makedirs(f'{out}/{name}', exist_ok=True)
    with open(f'{out}/{name}/ref.tsv', 'w') as f:
        for uid, d in picked:
            spk, ch, _ = uid.split('-')
            shutil.copy(f'{ROOT}/{spk}/{ch}/{uid}.flac', f'{out}/{name}/{uid}.flac')
            f.write(f'{uid}\t{d:.3f}\t{trans[uid]}\n')
    ds = [d for _, d in picked]
    print(name, len(picked), f'{min(ds):.2f}-{max(ds):.2f} s, mean {sum(ds)/len(ds):.2f}')
# b01: the ten shortest utterances in test-clean, speakers allowed to repeat (1.28-1.81 s).
os.makedirs(f'{out}/b01', exist_ok=True)
with open(f'{out}/b01/ref.tsv', 'w') as f:
    for uid, spk, d in sorted(utts, key=lambda x: x[2])[:10]:
        ch = uid.split('-')[1]
        shutil.copy(f'{ROOT}/{spk}/{ch}/{uid}.flac', f'{out}/b01/{uid}.flac')
        f.write(f'{uid}\t{d:.3f}\t{trans[uid]}\n')
