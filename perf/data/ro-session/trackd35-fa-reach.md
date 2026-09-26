<!-- ministral3-3b F16, RK3588 600 MHz, governor ondemand, taskset 0xf0 -t 4, one job in flight.
     llama-perplexity -c 2048 --chunks 2 -fa on, and llama-bench -p 2048 -n 0 -r 1 -v.
     Measured 2026-09-07 on libggml-rocket.so md5 5eaf4076a9f50dcb7e953156823d1d4e.
     Drivers: perf/data/trackd34-fa-reach.sh and perf/data/trackd35-reach2.sh.
     Raw: trackd34-fa-reach.log and trackd35-fa-reach.log beside this, md5 verified at both ends. -->

# The attention offload reaches llama-perplexity; only its summary line does not

`llama-perplexity` prints neither `ROCKET FA total` nor `ROCKET FA checksum` on the shapes
measured here, where `llama-bench` on the same model, `.so` and prompt length prints both. Two
causes look identical from outside: the scheduler places no `FLASH_ATTN_EXT` node on this backend
under that tool's graph, or the node runs and the summary is lost.

**It is the log.** The handler runs.

| arm | perplexity | chunk [1] |
|---|---:|---:|
| `ROCKET_FLASH_ATTN=1`, first process | **13.8994 +/- 0.82808** | 10.8620 |
| `ROCKET_FLASH_ATTN=1`, second process | **13.8994 +/- 0.82808** | 10.8620 |
| `ROCKET_FLASH_ATTN=0` | **13.9024 +/- 0.82829** | 10.8643 |

The two `=1` processes agree to every printed digit, so the stack is deterministic across
processes at a fixed thread count and the instrument can resolve a difference. The `=0` arm
differs, and it differs because the handler computes attention in fp16 through the NPU's QK and AV
where the CPU backend uses its own tiled kernel. **A knob that moved nothing would have printed
13.8994 three times.**

**Why the line is lost.** Both summaries are `GGML_LOG_INFO` emitted from an `atexit` handler, and
llama.cpp's `common_log` -- which `common_init` installs and `llama-bench` does not -- is
asynchronous and **discards messages once its worker is paused** (`common/log.cpp`, `add()`). The
handler's own arithmetic is unaffected.

**What follows.** Every published FA number is a `llama-bench` number, and that is a property of
where the probes can be read rather than of where the op runs, so the numbers are not scoped to
one harness. **The probe is scoped to one harness**: to read `ROCKET_FA_TIMING` or
`ROCKET_FA_CHECKSUM` under any tool built on `common_init`, expect nothing at exit and use
`llama-bench -v`.

**What this does not show.** One model and one shape. It says the op is placed under this tool at
`-c 2048` with `-fa on`; it does not survey `llama-server`, `llama-cli`, or a graph whose adjacent
matmuls the backend declines -- placement propagates to an FA node from an adjacent assigned op,
so it remains a property of the whole graph.

## Reading the driver version without dmesg

`ROCKET_DEBUG=1 ROCKET_LOG_LEVEL=debug ROCKET_LOG_STDERR=1` on a run that opens the device prints
`opened rocket: rocket 1.3.0`, taken from `DRM_IOCTL_VERSION` at open (`rocket_npu.c`). It depends
on no kernel log at all.

**Two conditions, and the first attempt met neither.** The prompt must be long enough to offload
something, since a run that opens no device never reaches the line: at `-p 64` nothing printed, at
`-p 2048` it did. And the level must admit DEBUG, since `ROCKET_LOG_STDERR` tees a line that was
emitted rather than raising the threshold itself. **The cost is 11825 rocket log lines** in that
run, so redirect it and grep rather than reading it.

The other two routes: `dmesg` reads the kernel ring buffer and rolls, and a board up six days had
lost the line entirely; `sudo journalctl -b -k | grep "Initialized rocket"` survives that and
returns every module load with a date, verified on this board at 2 days 11 h with a persistent
48.7 MB journal.
