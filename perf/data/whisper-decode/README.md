# Raw data for the Whisper decoder step on resident weights

Behind [perf/decode-gemv.md](../../decode-gemv.md) §"A Whisper decoder step on resident weights".
Turing RK1 on mainline `rocket` 1.3.0 at ~816 MHz, `ROCKET_KACC=1`, `taskset -c 4-7`, every CPU
policy on `performance` with `cpu-sleep` off, memory reset before every arm, 2026-10-10. The NPU
arm is `rocket-userspace` `tests/whisper_decode_step_probe.c`. The CPU arm is whisper.cpp v1.9.4
built with OpenMP, on its multilingual F16 models and `samples/jfk.wav`.

| file | what |
|---|---|
| `campaign-wall.py` | The step-time comparison: the probe's `step` mode against `whisper-cli -bs 1 -nf -t 4`, tiny, base and small, three passes with the arm order rotated |
| `campaign-wall.jsonl` | Its results, one record per arm and pass, with the SoC temperatures |
| `wdec_cpu.c` | The CPU instrument: `whisper_decode` timed per call through libwhisper, one encode, the clip's greedy token sequence replayed at each thread count in rotated order inside one process |
| `campaign-cpu.py` | The CPU comparison: the probe against `wdec_cpu` at 4, 2 and 1 threads and at 4 and 2 with `OMP_WAIT_POLICY=passive`, base and small, three passes |
| `campaign-cpu.jsonl` | Its results, one record per replay |

Both campaigns pin and restore the board through a script that writes the governor and idle
state with `sudo`, with the password taken from `SUDOPW` in the environment. The CPU column of
`campaign-wall.jsonl` is whole-process user+sys time, so its per-step CPU is a difference of two
runs and counts the mel spectrogram's extra frames. `wdec_cpu` measures the step directly, and the
note quotes it.
