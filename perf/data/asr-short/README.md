# Raw data for short-utterance STT

The record behind [perf/asr-cpu-relief.md](../../asr-cpu-relief.md#short-utterances-below-the-f16-floor),
"Short utterances below the F16 floor".

RK3588 (Turing RK1), mainline `rocket` 1.3.0 on kernel 7.2.8, NPU 600 MHz, CPU governor
`ondemand` (not pinned), `taskset -c 4-7`, 4 threads. whisper.cpp v1.9.4 `parakeet-cli` with
ggml-rocket `8b73e4c` built against its ggml, and Parakeet TDT 0.6B v3 F16 in whisper.cpp's own
`.bin` format. llama.cpp b11242 for the LLM runs. Measured 2026-09-30 UTC.

## Files

| file | what |
|---|---|
| `mkbuckets.py` | cuts the six LibriSpeech test-clean length sets, ten utterances each; `ref-b*.tsv` are its output |
| `campaign1.sh` | five arms x six lengths x a 1-file and a 10-file process x 3 rotated passes |
| `raw-run1.tsv` | one row per `campaign1.sh` process: wall, user, sys, user-mode cycles, transcript md5; pass `w0` is the discarded warm-up |
| `analyze.py`, `summary-run1.txt` | the per-utterance cost, (10-file - 1-file) / 9 per pass, its medians, WER and transcript identity |
| `campaign3.sh`, `admission-profile.log` | which GEMMs each floor admits per length (`calls` counts offloaded GEMMs), and `ROCKET_MM_PROFILE` per 10-utterance process |
| `campaign2.sh`, `neighbor-nice0.log` | a b02 stream beside `llama-bench` decode of Llama-3.2-3B `Q4_K_M` on the same cores, 3 passes |
| `campaign6b.sh`, `neighbor-nice19.log` | the same with the neighbor at `nice -n 19`, 2 passes, an 800-utterance stream |
| `campaign4.sh`, `llm-crossover.log` | Llama-3.2-3B F16 `llama-bench -p 16,32,64,128`, NPU at floor 4 against the CPU, 2 passes |
| `campaign5.sh`, `sherpa_bench.py` | sherpa-onnx 1.13.8, Parakeet TDT 0.6B v2 and v3 int8, one persistent process, 3 passes |
| `analyze2.py` | reads the neighbor, LLM and sherpa records |

## Arms

| arm | environment |
|---|---|
| `cpu` | `ROCKET_MIN_M=1000000 ROCKET_FLASH_ATTN=0`: the backend loads and declines every op |
| `cpup` | `cpu` plus `OMP_WAIT_POLICY=PASSIVE` |
| `def` | the defaults, `ROCKET_MIN_M=128` |
| `forced` | `ROCKET_MIN_M=4` |
| `forcedp` | `forced` plus `OMP_WAIT_POLICY=PASSIVE` |

`cpu_s` is the process's `user + sys`, the core-seconds this record is about. `gcycles_u` is
user-mode cycles on both clusters from `perf stat`, which counts no kernel time. In the neighbor
logs, `stt_files_during` is the utterances the stream finished while the neighbor ran, and
`stt_alive=1` says the stream was still running when the neighbor ended.

## Known limits

The governor was `ondemand` throughout, because `sudo` was not available that session. It is a
bias toward the CPU arm on an offloading process. The sherpa-onnx per-utterance walls are bimodal
under it: the same 4.4-5.9 s utterance read 0.53 s in one pass and 1.02 s in another. So the sherpa
medians mix a fast and a slow mode. The `nice -n 19` forced-arm windows are 8-10 utterances each.
