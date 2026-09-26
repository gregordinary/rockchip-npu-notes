# Raw data for [perf/asr-streaming-service.md](../../asr-streaming-service.md)

RK3588 (Turing RK1), mainline `rocket` 1.3.0, NPU 600 MHz, CPU governor `performance`, the server
pinned `taskset -c 4-7` with 4 threads unless the arm says otherwise, `ROCKET_KACC=1`, whisper.cpp
master `eacbd82`, `ggml-small` (multilingual). Every arm is one `whisper-server` process driven the
way OpenWebRX+ drives it: one multipart `file` field holding a 12 kHz WAV, one request at a time.
The server is billed per request by its own `utime + stime`, read from `/proc/PID/stat` before and
after the request, so the model load is outside every number. Arms rotate across passes with a
memory reset before each arm. 2026-09-03.

| file | what |
|---|---|
| `campaign.tsv` | The lever sweep, 11 arms x 3 channel conditions x 2 passes, 20 s chunks (30 s for the `c30` arms) |
| `campaign2.tsv` | The no-fallback arm (per-request `temperature_inc=0`), the 25 s chunks, and the stacked `-ac 1000` configurations |
| `campaign3.tsv` | The headroom context (`-ac 1200`), the 20 s and 25 s `-nt` stacks, and the `-sns` probe |
| `campaign4.tsv` | The three surviving configurations on the second chapter (`s121-127105`), two passes |
| `campaign5.tsv` | The defaults and the 20 s stack under `ondemand`, unpinned, one pass |
| `campaign-vendor.tsv` | The shipped arm, the CPU arm and the 20 s and 30 s recommendations on a vendor-kernel RK1 (`6.1.172-vendor-rk35xx`, `rknpu` 0.9.8, through rknpu-submit), NPU 1000 MHz and DDR 2112 MHz pinned, two passes, 2026-09-26 |
| `wer-campaign*.json` | Word error rate per arm, pass, chapter and condition against the LibriSpeech references, one file per sweep |
| `svc-campaign.sh` / `svc-campaign2.sh` | The arm tables; `ARMS`, `PASSES` and the stream are the knobs |
| `svc-campaign-vendor.sh` / `svc-arm-vendor.sh` | The vendor-board run: the same arms, the server run as the invoking user, the governors pinned and restored |
| `svc-arm.sh` | One arm: start the server under this arm's environment and flags, drive it, stop it |
| `svc-client.py` | The client: posts chunks like `owrx/transcribe.py`, bills the server PID, keeps the text |
| `mkradio.py` | Builds the chunk sets from LibriSpeech `test-clean`: one chapter per stream, cut at 20 s and 30 s, through an SSB-shaped channel |
| `analyze.py` | Per-arm core-seconds per audio-second, wall, realtime, and pass-paired ratios |
| `wer.py` | The scorer: lowercase, punctuation stripped, small integers spelled out, Levenshtein over words |
| `whisper-server-no-fallback.patch` | The one-line server fix that makes `-nf` do what `whisper-cli`'s does |

`cpu_s` is the server's CPU core-seconds for the request, the number this work is about; `wall_s`
is the request latency and is not a substitute. `audio_s` is the chunk's length, so `cpu_s /
audio_s` summed over a set is the cost per second of channel time while it is busy.

The channel conditions are `clean` (the stream resampled to 12 kHz), `ssb15` (band-pass
300-2700 Hz, noise at 15 dB SNR over the speech-active RMS, +25 Hz carrier offset, 3 dB fading,
soft limiting) and `ssb5` (5 dB, +40 Hz, 6 dB). They are a radio-shaped proxy with a known
transcript, not radio: what they cannot certify is real fading, real compression and real accents,
so the WER column ranks configurations against each other rather than predicting a channel.
