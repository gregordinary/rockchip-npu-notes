<!-- gemma4-12b F16 100%-resident probe at -p 2048   TESTS='-p 2048 -n 0 -r 2'
     gguf=<data>/gemma4/gemma-4-12b-it-F16.gguf (23832065184 bytes)
     so=61f02a02345f3b2e27cef45de8ade0c2  bin=4777f8f8d193e63502d3db3b9c028fb6
     MemTotal=32547712 kB  SwapTotal=0 kB
     clk=600000000 Hz
     dmesg OOM lines before start: 3
-->

[2026-09-01T20:52:21Z] res100 probe start

### res100  env='ROCKET_N_THREADS=8 ROCKET_QUANT_RESIDENT_RESERVE_MB=6144'  20:52:21
    MemAvailable before: 31000 MB
    rc=0
    MemFree low-water: 260 MB   MemAvailable low-water: 5521 MB   samples=202
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 24484MB (MemAvailable 30628MB - reserve 6144MB, no swap)
    [f16-resident] weights offered to the resident route: 328 resident on the NPU (20790MB), 0 streamed via the per-call pack -- 100% resident
    ROCKET profile total(ms): pack=108600 (packA=28064 packB=12011) gen=5041 sync=18653 submit=3666 wait=1116178 read=42329  over 33792 job-batches
    --- llama-bench row ---
    | model                          |       size |     params | backend    | ngl |            test |                  t/s |
    | ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
    | gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         20.99 ± 1.20 |

[2026-09-01T20:59:06Z] ALL-DONE
