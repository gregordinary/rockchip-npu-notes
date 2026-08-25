# Cut fixed-length clips from jfk_long.wav for the M6 length sweep.
# All clips start at the same offset so they share a speech onset; each longer
# clip is a strict superset of the shorter ones, which keeps the length axis the
# only thing that varies.
import os, wave, sys

SRC = os.environ.get("ASR_SRC", "/path/to/asr-workdir/jfk_long.wav")
START_S = 4.0
LENGTHS = [3, 10, 30, 60, 120]

w = wave.open(SRC)
sr = w.getframerate()
w.setpos(int(START_S * sr))
raw = w.readframes(int(max(LENGTHS) * sr))

for n in LENGTHS:
    out = wave.open(os.path.join(os.path.dirname(SRC), "clips", "clip_%03ds.wav" % n), "wb")
    out.setnchannels(1); out.setsampwidth(2); out.setframerate(sr)
    out.writeframes(raw[: n * sr * 2])
    out.close()
    print("clip_%03ds.wav" % n, n * sr, "frames")
