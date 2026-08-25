# Raw data for [perf/asr-cpu-relief.md](../../asr-cpu-relief.md)

RK3588 (Turing RK1), vendor `rknpu 0.9.8` on Armbian `6.1.115-vendor-rk35xx`, NPU 600 MHz, A76
cluster pinned `performance`, `taskset -c 4-7`, 4 threads, `ROCKET_KACC=1`. Every sweep discards
a warm-up run per cell and alternates the arm order between repetitions. 2026-08-24.

| file | what |
|---|---|
| `sweep-whisper.tsv` | whisper.cpp, 3 models x 5 lengths x 2 arms x 3 reps, clean read speech |
| `sweep-aunt.tsv` | the same on hard conversational speech, 2 reps — the clip set the transcript gate actually discriminates on |
| `sweep-transcribe.tsv` | transcribe.cpp, SenseVoice + Parakeet-CTC Q8_0, 2 arms |
| `sweep-transcribe-q.tsv` | the same with a third arm at `ROCKET_MIN_M_QUANT=128` |
| `sweep-whisper.sh` / `sweep-transcribe.sh` | the harnesses; `CLIPPFX`, `MODELS`, `CLIPS`, `REPS` are the knobs |
| `weight-pack-cap.sh` | the `packB`-share measurement that closed the weight-cache lever |
| `mkclips.py` | cuts the length ladder so each longer clip is a strict superset of the shorter ones |

`cpu_s` is `user + sys` of the child process — the CPU core-seconds, which is the number this
work is about; `wall_s` is not a substitute for it. `md5` is of the whitespace-stripped
transcript, so equal md5 across the two arms is the faithfulness gate.
