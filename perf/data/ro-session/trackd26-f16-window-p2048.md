<!-- gemma4-12b F16 residency admission readout   PROMPT='-p 2048 -n 0 -r 1'
     gguf=<data>/gemma4/gemma-4-12b-it-F16.gguf (23832065184 bytes)
     so=61f02a02345f3b2e27cef45de8ade0c2
     MemTotal=32547712 kB  SwapTotal=0 kB
     dmesg OOM lines before start: 3
-->

[2026-09-03T01:35:11Z] readout start, 4 arms

### t5_a  env=''  01:35:11
    MemAvailable before: 30914 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 20991MB (MemAvailable 30526MB - reserve 9535MB, no swap)
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [f16-resident] admission first declined at 18078MB resident: MemAvailable 9401MB fell below the 9535MB reserve floor

### t5_b  env=''  01:40:18
    MemAvailable before: 30924 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21017MB (MemAvailable 30552MB - reserve 9535MB, no swap)
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [f16-resident] admission first declined at 18078MB resident: MemAvailable 9395MB fell below the 9535MB reserve floor

### t6_a  env='ROCKET_N_THREADS=6'  01:45:34
    MemAvailable before: 30932 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21026MB (MemAvailable 30561MB - reserve 9535MB, no swap)
    [f16-resident] weights offered to the resident route: 284 resident on the NPU (17853MB), 44 streamed via the per-call pack -- 87% resident
    [f16-resident] admission first declined at 17853MB resident: MemAvailable 9527MB fell below the 9535MB reserve floor

### t6_b  env='ROCKET_N_THREADS=6'  01:50:36
    MemAvailable before: 30908 MB
    rc=0
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21007MB (MemAvailable 30542MB - reserve 9535MB, no swap)
    [f16-resident] weights offered to the resident route: 284 resident on the NPU (17853MB), 44 streamed via the per-call pack -- 87% resident
    [f16-resident] admission first declined at 17853MB resident: MemAvailable 9532MB fell below the 9535MB reserve floor

[2026-09-03T01:55:48Z] ALL-DONE
