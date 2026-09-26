<!-- gemma4-12b F16 per-call weight-pack CAP readout   TESTS='-p 2048 -n 0 -r 2'
     gguf=<data>/gemma4/gemma-4-12b-it-F16.gguf (23832065184 bytes)
     so=61f02a02345f3b2e27cef45de8ade0c2  bin=4777f8f8d193e63502d3db3b9c028fb6
     MemTotal=32547712 kB  SwapTotal=0 kB
     clk=600000000 Hz
     gov=performance
     dmesg OOM lines before start: 3
-->

[2026-09-01T20:43:32Z] packcap readout start

### res87  env=''  20:43:32
    MemAvailable before: 30973 MB
    rc=0
    MemFree low-water: 259 MB   MemAvailable low-water: 7719 MB   samples=81
    [rocket] ROCKET_F16_RESIDENT=auto -> resident budget 21083MB (MemAvailable 30619MB - reserve 9535MB, no swap)
    [f16-resident] weights offered to the resident route: 286 resident on the NPU (18078MB), 42 streamed via the per-call pack -- 87% resident
    [f16-resident] admission first declined at 18078MB resident: MemAvailable 9475MB fell below the 9535MB reserve floor
    ROCKET profile total(ms): pack=143116 (packA=21006 packB=88259) gen=4003 sync=15244 submit=2668 wait=655085 read=34050  over 21120 job-batches
    --- llama-bench row ---
    | model                          |       size |     params | backend    | ngl |            test |                  t/s |
    | ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
    | gemma4 ?B F16                  |  22.18 GiB |    11.91 B | ROCKET     |  -1 |          pp2048 |         19.94 ± 2.23 |

[2026-09-01T20:50:19Z] ALL-DONE
