<!-- gemma4-12b F16 residency admission readout   PROMPT='-p 512 -n 0 -r 1'
     gguf=<data>/gemma4/gemma-4-12b-it-F16.gguf (23832065184 bytes)
     so=61f02a02345f3b2e27cef45de8ade0c2
     MemTotal=32547712 kB  SwapTotal=0 kB
     dmesg OOM lines before start: 3
-->

[2026-09-02T03:48:42Z] readout start, 3 arms

### t6_b  env='ROCKET_N_THREADS=6'  03:48:43
    MemAvailable before: 30963 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21217MB (MemAvailable 30752MB - reserve 9535MB, no swap)
    [f16-resident] weights offered to the resident route: 293 resident on the NPU (18506MB), 35 streamed via the per-call pack -- 89% resident
    [f16-resident] admission first declined at 18506MB resident: MemAvailable 9299MB fell below the 9535MB reserve floor

### t7_b  env='ROCKET_N_THREADS=7'  03:55:00
    MemAvailable before: 30936 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21222MB (MemAvailable 30758MB - reserve 9535MB, no swap)
    [f16-resident] weights offered to the resident route: 274 resident on the NPU (17302MB), 54 streamed via the per-call pack -- 84% resident
    [f16-resident] admission first declined at 17302MB resident: MemAvailable 9448MB fell below the 9535MB reserve floor

### t6_c  env='ROCKET_N_THREADS=6'  04:01:02
    MemAvailable before: 30937 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21226MB (MemAvailable 30762MB - reserve 9535MB, no swap)
    [f16-resident] weights offered to the resident route: 293 resident on the NPU (18506MB), 35 streamed via the per-call pack -- 89% resident
    [f16-resident] admission first declined at 18506MB resident: MemAvailable 9358MB fell below the 9535MB reserve floor

[2026-09-02T04:07:15Z] ALL-DONE
