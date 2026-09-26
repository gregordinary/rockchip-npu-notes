<!-- gemma4-12b F16 residency admission readout   PROMPT='-p 512 -n 0 -r 1'
     gguf=<data>/gemma4/gemma-4-12b-it-F16.gguf (23832065184 bytes)
     so=61f02a02345f3b2e27cef45de8ade0c2
     MemTotal=32547712 kB  SwapTotal=0 kB
     dmesg OOM lines before start: 3
-->

[2026-09-01T12:54:07Z] readout start, 2 arms

### joint_t8_r6144  env='ROCKET_N_THREADS=8 ROCKET_QUANT_RESIDENT_RESERVE_MB=6144'  12:54:07
    MemAvailable before: 31002 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 24491MB (MemAvailable 30635MB - reserve 6144MB, no swap)
    [f16-resident] weights offered to the resident route: 328 resident on the NPU (20790MB), 0 streamed via the per-call pack -- 100% resident

### joint_t8_r6144_again  env='ROCKET_N_THREADS=8 ROCKET_QUANT_RESIDENT_RESERVE_MB=6144'  12:56:45
    MemAvailable before: 30996 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 24472MB (MemAvailable 30616MB - reserve 6144MB, no swap)
    [f16-resident] weights offered to the resident route: 328 resident on the NPU (20790MB), 0 streamed via the per-call pack -- 100% resident

[2026-09-01T12:59:16Z] ALL-DONE
