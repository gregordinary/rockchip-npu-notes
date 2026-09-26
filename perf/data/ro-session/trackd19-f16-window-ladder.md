<!-- gemma4-12b F16 residency admission readout   PROMPT='-p 512 -n 0 -r 1'
     gguf=<data>/gemma4/gemma-4-12b-it-F16.gguf (23832065184 bytes)
     so=61f02a02345f3b2e27cef45de8ade0c2
     MemTotal=32547712 kB  SwapTotal=0 kB
     dmesg OOM lines before start: 3
-->

[2026-09-02T03:10:17Z] readout start, 6 arms

### t5_a  env=''  03:10:17
    MemAvailable before: 30093 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21028MB (MemAvailable 30563MB - reserve 9535MB, no swap)
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [f16-resident] admission first declined at 18078MB resident: the NPU IOVA window filled (raise ROCKET_N_THREADS for more fds)
    ALLOC: ROCKET_CREATE_BO(23625728): No space left on device

### t5_b  env=''  03:20:28
    MemAvailable before: 30948 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21186MB (MemAvailable 30722MB - reserve 9535MB, no swap)
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [f16-resident] admission first declined at 18078MB resident: the NPU IOVA window filled (raise ROCKET_N_THREADS for more fds)
    ALLOC: ROCKET_CREATE_BO(23625728): No space left on device

### t6  env='ROCKET_N_THREADS=6'  03:22:50
    MemAvailable before: 30924 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21223MB (MemAvailable 30759MB - reserve 9535MB, no swap)
    [f16-resident] weights offered to the resident route: 293 resident on the NPU (18506MB), 35 streamed via the per-call pack -- 89% resident
    [f16-resident] admission first declined at 18506MB resident: MemAvailable 9354MB fell below the 9535MB reserve floor

### t7  env='ROCKET_N_THREADS=7'  03:29:19
    MemAvailable before: 30927 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21203MB (MemAvailable 30738MB - reserve 9535MB, no swap)
    [f16-resident] weights offered to the resident route: 274 resident on the NPU (17302MB), 54 streamed via the per-call pack -- 84% resident
    [f16-resident] admission first declined at 17302MB resident: MemAvailable 9444MB fell below the 9535MB reserve floor

### t8  env='ROCKET_N_THREADS=8'  03:34:50
    MemAvailable before: 30947 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21219MB (MemAvailable 30754MB - reserve 9535MB, no swap)
    [f16-resident] weights offered to the resident route: 287 resident on the NPU (18191MB), 41 streamed via the per-call pack -- 88% resident
    [f16-resident] admission first declined at 18191MB resident: MemAvailable 9409MB fell below the 9535MB reserve floor

### t8_r6144  env='ROCKET_N_THREADS=8 ROCKET_QUANT_RESIDENT_RESERVE_MB=6144'  03:41:31
    MemAvailable before: 30931 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 24615MB (MemAvailable 30759MB - reserve 6144MB, no swap)
    [f16-resident] weights offered to the resident route: 328 resident on the NPU (20790MB), 0 streamed via the per-call pack -- 100% resident

[2026-09-02T03:48:26Z] ALL-DONE
