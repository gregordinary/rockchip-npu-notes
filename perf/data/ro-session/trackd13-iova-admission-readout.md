<!-- gemma4-12b F16 residency admission readout   PROMPT='-p 512 -n 0 -r 1'
     gguf=<data>/gemma4/gemma-4-12b-it-F16.gguf (23832065184 bytes)
     so=61f02a02345f3b2e27cef45de8ade0c2
     MemTotal=32547712 kB  SwapTotal=0 kB
     dmesg OOM lines before start: 3
-->

[2026-09-01T12:42:13Z] readout start, 4 arms

### auto_default  env=''  12:42:13
    MemAvailable before: 31007 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21095MB (MemAvailable 30631MB - reserve 9535MB, no swap)
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [f16-resident] admission first declined at 18078MB resident: the NPU IOVA window filled (raise ROCKET_N_THREADS for more fds)
    ALLOC: ROCKET_CREATE_BO(23625728): No space left on device

### nthreads8  env='ROCKET_N_THREADS=8'  12:44:52
    MemAvailable before: 31020 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21099MB (MemAvailable 30634MB - reserve 9535MB, no swap)
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [f16-resident] admission first declined at 18078MB resident: MemAvailable 9469MB fell below the 9535MB reserve floor

### reserve7168  env='ROCKET_QUANT_RESIDENT_RESERVE_MB=7168'  12:47:30
    MemAvailable before: 31008 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 23466MB (MemAvailable 30634MB - reserve 7168MB, no swap)
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [f16-resident] admission first declined at 18078MB resident: the NPU IOVA window filled (raise ROCKET_N_THREADS for more fds)
    ALLOC: ROCKET_CREATE_BO(23625728): No space left on device

### reserve6144  env='ROCKET_QUANT_RESIDENT_RESERVE_MB=6144'  12:50:11
    MemAvailable before: 31014 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 24494MB (MemAvailable 30638MB - reserve 6144MB, no swap)
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [f16-resident] admission first declined at 18078MB resident: the NPU IOVA window filled (raise ROCKET_N_THREADS for more fds)
    ALLOC: ROCKET_CREATE_BO(23625728): No space left on device

[2026-09-01T12:52:49Z] ALL-DONE
