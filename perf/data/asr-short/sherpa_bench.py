# sherpa-onnx Parakeet TDT int8 CPU baseline. One persistent process,
# model loaded once, per-utterance wall and process CPU (utime+stime over all threads).
import sys, time, resource, wave, glob, os, json
import numpy as np, sherpa_onnx
mdir, clipdir, threads, passes = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
p = lambda n: glob.glob(f'{mdir}/{n}')[0]
t0 = time.perf_counter()
rec = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=p('encoder*.onnx'), decoder=p('decoder*.onnx'), joiner=p('joiner*.onnx'),
    tokens=p('tokens.txt'), num_threads=threads, model_type='nemo_transducer', provider='cpu')
load_s = time.perf_counter() - t0
def load(f):
    with wave.open(f) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        return w.getframerate(), a
def cpu(): r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
def once(f):
    sr, a = load(f); s = rec.create_stream(); s.accept_waveform(sr, a)
    w0, c0 = time.perf_counter(), cpu(); rec.decode_stream(s); w1, c1 = time.perf_counter(), cpu()
    return w1 - w0, c1 - c0, s.result.text
buckets = sorted(glob.glob(f'{clipdir}/b*'))
once(sorted(glob.glob(f'{buckets[0]}/*.wav'))[0])            # warm-up, discarded
print(json.dumps({'load_s': load_s, 'threads': threads, 'model': os.path.basename(mdir)}))
for ps in range(passes):
    for b in buckets:
        for f in sorted(glob.glob(f'{b}/*.wav')):
            w, c, txt = once(f)
            print(json.dumps({'pass': ps, 'bucket': os.path.basename(b), 'utt': os.path.basename(f)[:-4], 'wall': w, 'cpu': c, 'text': txt}), flush=True)
