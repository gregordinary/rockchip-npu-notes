# Raw data for the same-board RKNN encoder comparison

Behind [encodings/whisper-encoder.md](../../../encodings/whisper-encoder.md) §"In-model fused
integration" and the summary in [whisper-encoder.md](../whisper-encoder.md). Turing RK1 on the
vendor kernel, `6.1.172-vendor-rk35xx`, `rknpu` 0.9.8, 2026-09-26.

The RKNN model is `whisper_encoder_base_20s.rknn` from the Hugging Face repository
`harvestsu/whisper-edge` at `e318664`: `rknn_model_zoo`'s fp16 whisper-base encoder, compiled
by rknn-toolkit2 2.3.0 for the rk3588, input `[1,80,2000]`, output `[1,1000,512]`. The runtime
is Rockchip's `librknnrt` from `airockchip/rknn-toolkit2`, 2.3.2 unless a file says 2.3.0.
Neither is redistributed here.

| file | what |
|---|---|
| `rknn_enc_bench.c` | The timing harness: `rknn_run` alone per call, median of the warm calls, plus the set-run-get time and the first call's output statistics |
| `rknn_enc_perf.c` | Attribution only: the runtime's `RKNN_QUERY_PERF_RUN` and per-op `RKNN_QUERY_PERF_DETAIL`, with perf collection on |
| `enc-campaign.sh` | The five-pass comparison: RKNN on cores 0-2 and on core 0, the ggml-rocket drop-in and the CPU, at 1000 and 600 MHz |
| `campaign.tsv` | Its results, one row per arm, clock and pass, with the NPU clock read back and the SoC temperatures |
| `clock-witness.sh` | 1000, 300, 600 and 1000 MHz on both core masks, a timed arm then a perf-detail arm |
| `perf-detail-<MHz>-<mask>.txt` | The per-op tables it wrote, one per clock and core mask |
| `rt-ab.sh` | `librknnrt` 2.3.2 against 2.3.0, alternated at 1000 MHz |
| `perf-detail-1000-012-rt<version>.txt` | The per-op table under each runtime |
| `dmc-campaign.sh` | `dmc_ondemand` against DDR pinned at 2112 MHz, three alternating passes, DDR sampled every 50 ms |
| `dmc.tsv` | Its results, with each arm's DDR-frequency histogram |

Every script pins the CPU governor to `performance` and the NPU clock through its devfreq node,
reads each pin back and aborts if it did not take, and restores the original governors on any
exit. Sysfs writes go through `sudo`, with the password taken from `PW` in the environment.
Build the harnesses against `rknn_api.h` and `librknnrt.so` from the same runtime release:

```sh
gcc -O2 -o rknn_enc_bench rknn_enc_bench.c -I. -L. -lrknnrt -lm -Wl,-rpath,'$ORIGIN'
gcc -O2 -o rknn_enc_perf rknn_enc_perf.c -I. -L. -lrknnrt -Wl,-rpath,'$ORIGIN'
```

`librknnrt` opens the NPU's DRM card node, `root:video` on this image, so the account needs
`video` as well as the `render` group ggml-rocket's provider uses.
