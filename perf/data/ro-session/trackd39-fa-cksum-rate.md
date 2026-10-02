<!-- The FA output checksum's disagreement rate, and the op it starts at. RK3588 RK1, rocket 1.3.0,
     600 MHz, governor ondemand, power/control=auto, board otherwise idle. gemma4-12b F16,
     llama-bench -p 2048 -n 0 -r 1 -t 4 -v under taskset 0xf0, ROCKET_KACC=1,
     ROCKET_FA_THREADS=1, ROCKET_FA_CHECKSUM=2, page cache dropped before every run, one process
     per run. libggml-rocket.so 138de09d05f235f8c50c7d9d7df3eaa5, librocketnpu.a
     1fd95700abdad3143afdd04c6458b5f1. Measured 2026-09-26 00:33-02:00 UTC.
     Raw: trackd39-fa-cksum-rate.log. Harness: ../trackd39-fa-cksum-rate.sh. -->

# The FA handler's output takes one of two values, and the disagreement starts at one op

Twenty runs at one setting print one of two aggregate hashes: `8852206bf3ccda62` fifteen times
and `345a87c827fa972b` five times. The second is the odd hash the 2026-09-07 runs drew on another
build, `c66ef3e8`. Every run reports 288 ops, so every run ran the same graph.

## The rate

| Campaign | `.so` | Runs | Odd runs |
|---|---|---:|---:|
| 2026-09-07 gate and reproduction | `c66ef3e8` | 7 | 2 |
| 2026-09-07 localization | `138de09d` | 6 | 0 |
| This campaign | `138de09d` | 20 | 5 |

**7 of 33, about one run in five, and only two output states in 33.** The odd runs here were 9, 10,
11, 13 and 16, so they cluster rather than spread evenly. Twenty runs are too few to say whether
that clustering is a board-state regime or chance.

## The op

Every odd run differs from every modal run in the same seven per-op lines, and the five odd runs
agree with each other on all 288. The 288 ops are two passes of 144, llama-bench's warm-up and its
timed rep. Each pass has four microbatches of 48 layers, and offloads the second to the fourth, at
`n_kv` 1024, 1536, and 1536 or 2048. The first is below `MIN_KV`.

| Op | Pass | Microbatch | Layer |
|---:|---|---:|---:|
| 189 | timed | 2 | 45 |
| 190, 191 | timed | 2 | 46, 47 |
| 238, 239 | timed | 3 | 46, 47 |
| 286, 287 | timed | 4 | 46, 47 |

That set is exactly what reads op 189's output. Layers 46 and 47 of the same microbatch take it
through the residual stream. The same two layers in later microbatches read the K and V those
layers cached from it. Layer 45 of the later microbatches reads its own cache, computed before op
189 ran, and does not differ. So **the disagreement starts at op 189 and the other six ops are
its dataflow** [inferred from the layer structure, every op between them being identical].

It follows that op 189's difference changes values later layers consume. So it is not confined to
padding, though how large it is remains unknown.

## What this could not see

- **The magnitude.** A hash moves on one bit as readily as on a surface. A value dump of op 189 in
  one modal and one odd run is the measurement.
- **Whether op 189's inputs already differ.** Level 2 hashes the handler's output only. Hashing
  each op's Q, K, V and mask as well would say whether the difference is upstream (a matmul feeding
  layer 45) or in the handler's own arithmetic.
- **Why that op.** The same op in the warm-up pass never differed. A run-time-dependent route (the
  residency latch on `MemAvailable`, core placement across three cores) is a candidate
  [hypothesis], and the conditions differ from the 2026-09-07 runs, which pinned the governor and
  `power/control`.
