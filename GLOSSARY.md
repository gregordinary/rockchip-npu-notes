# Glossary

The terms these notes use that Rockchip's and NVIDIA's documentation does not define, or defines
in another sense. They are the NPU's block and stage names, which the TRM gives without
expanding most of them, the notes' own terms, and the abbreviations and terms of the models the
notes run. Register and field names are not listed: the note for each block gives them, and
[hardware-overview.md](hardware-overview.md) maps the blocks.

An abbreviation's definition opens with its expansion, or with "Not expanded." where its
source gives none. The Source column names the document that defines the term, or says Coined
for a term these notes introduce. The last column gives a coined term's nearest standard term,
and a Rockchip term's NVDLA counterpart.

## Terms

| Term | Definition | Source | Nearest standard term |
|---|---|---|---|
| ASR | Automatic speech recognition. The transcription of speech audio to text | Field | |
| Atom | The 16 bytes of channels at one position of a cube: the unit in which the DPU reads and writes the cube | NVDLA | |
| Backstop | The job timeout of `rocket`: the time after a kick at which the driver retires a job that has not completed ([chips/rk3576.md](chips/rk3576.md)) | Coined | Job timeout |
| Bit-exact | Equal, bit for bit, to a reference computed on the host | Field | |
| Blind settle | The fixed wait after `PC_DONE` that `dpu_blind_us` sets. At its end, `rocket` signals the fence of a job flagged `ROCKET_JOB_NO_DPU_DONE`, whatever the program is still doing ([chips/rk3576.md](chips/rk3576.md)) | Coined | |
| BN stage | Not expanded. The second of the DPU's three affine stages, after the BS stage and before the EW stage. NRDMA reads its operands from memory ([encodings/sdp-stage-precision.md](encodings/sdp-stage-precision.md)) | RK3588 TRM | NVDLA SDP X2 |
| BO | Buffer object. A buffer allocated with `CREATE_BO` and mapped into the NPU's address space | Linux kernel | |
| Bootstrap | The two device passes in which the RK3576 W8A8 path of `ggml-rocket` estimates each output column's accumulator maximum from the int8 matmul's own output ([perf/rk3576-w8a8-llm.md](perf/rk3576-w8a8-llm.md)) | Coined | |
| Bracket | A `PREP_BO` and `FINI_BO` pair around a CPU access to a BO. `PREP_BO` syncs the BO for the CPU, and `FINI_BO` syncs it back for the device ([perf/bo-sync-cost.md](perf/bo-sync-cost.md)) | Coined | |
| BRDMA | B read DMA, with the B not expanded. The DPU_RDMA channel that reads the BS stage's operands from memory | RK3588 TRM, NVDLA | |
| BS stage | Not expanded. The first of the DPU's three affine stages, which also contains the CPEND term. BRDMA reads its per-channel operands from the coefficient buffer ([encodings/sdp-stage-precision.md](encodings/sdp-stage-precision.md)) | RK3588 TRM | NVDLA SDP X1 |
| C2 | The number of channels stored together at each position of a cube layout. It is 16 bytes divided by the element size: 8 at fp16 and 16 at int8. It is the C2 of RKNN's NC1HWC2 layout ([encodings/tile-layouts.md](encodings/tile-layouts.md)) | RKNN Toolkit2 | NVDLA atomic channel count |
| Calibration forward | One of the first forward passes, `ROCKET_RK3576_NCAL` of them, in which the RK3576 W8A8 path runs the bootstrap on every weight. After the last one, the path freezes each output column's scale ([perf/rk3576-w8a8-llm.md](perf/rk3576-w8a8-llm.md)) | Coined | |
| CALMAP | Not expanded. The per-column calibration readout of the RK3576 W8A8 path of `ggml-rocket`, a CSV file that `ROCKET_RK3576_CALMAP` names. It has one row per output column per calibration forward ([perf/rk3576-w8a8-llm.md](perf/rk3576-w8a8-llm.md)) | Coined | |
| CBUF | Convolution buffer. The on-chip buffer of an NPU core that holds the feature data and weights a task reads. Its size and bank count are machine parameters ([chips/README.md](chips/README.md)) | NVDLA | |
| CBUF weight streaming | A matmul task form in which the weight cube is larger than the CBUF weight banks the task programs. The input tile stays in the data banks, and the CNA cycles the 32-kernel weight groups through the weight banks ([matmul-as-conv.md](matmul-as-conv.md)) | Coined | |
| Chained run | A maximal sequence of RK3576 layers whose register programs execute in one kick. Every layer but the last leaves a cube, and every layer but the first reads a cube that an earlier member wrote ([chips/rk3576.md](chips/rk3576.md)). A member is a convolution, a pool, or a placed concatenation, which has no register program | Coined | |
| Clearing idle | The idle interval after which the NPU power domain suspends, which clears the wide-output hazard: twice `power/autosuspend_delay_ms` ([chips/rk3576.md](chips/rk3576.md)) | Coined | |
| Clip fraction | The share of output columns, counted once per calibration forward, whose output saturates in the bootstrap's second pass. A column saturates when its largest int8 code reaches 127 ([perf/rk3576-w8a8-llm.md](perf/rk3576-w8a8-llm.md)) | Coined | Clipping rate |
| Clock park | The rate to which the NPU clock returns while the NPU is idle ([perf/clock.md](perf/clock.md)) | Coined | |
| CNA | Not expanded. The convolution front end of an NPU core (registers `0x1xxx`). It loads feature data and weights into the CBUF and feeds them to the MAC array. Some third-party code expands the name as "Convolution Neural Accelerator", which no Rockchip document gives | RK3588 TRM | NVDLA CDMA and CSC |
| Coefficient buffer | The buffer in memory that holds the BS stage's per-output-channel operands, organized in coefficient groups, which BRDMA reads ([encodings/out-cvt-converter.md](encodings/out-cvt-converter.md)) | Coined | |
| Coefficient group | One entry of the coefficient buffer: 64 bytes, which hold the operands of 8 output channels. Each channel has an int32 bias `A`, an int16 CPEND operand `B` and an int16 multiplier `C`. The RK3576's depthwise path uses a shorter group ([chips/rk3576-regcmd.md](chips/rk3576-regcmd.md)) | Coined | |
| CORE | The block of an NPU core that holds the MAC array and its accumulators (registers `0x3xxx`). The TRM also writes "NPU CORE" for a whole core, and these notes write CORE for the block and "core" for the core | RK3588 TRM | NVDLA CMAC and CACC |
| CPEND | Not expanded. A BS-stage term that adds an operand times the window sum of the CNA's input, border pad included. Its operand is `DPU_BS_OW_OP` per tensor or the coefficient group's `B` per channel, and `DPU_BS_OW_CFG.OD_BYPASS` bypasses it ([depthwise-conv.md](depthwise-conv.md)) | RK3588 TRM | |
| CTC | Connectionist temporal classification. A speech model's output head that emits a token or a blank for each audio frame, so that the transcript needs no autoregressive decoder | Field | |
| Cube join | A connection between two RK3576 layers in which the consumer reads the producer's output BO as its feature cube. The consumer need not be the next layer. No de-tile, pack or cache maintenance runs between them ([chips/rk3576.md](chips/rk3576.md)) | Coined | |
| Cube layout | The tiled memory layout in which the CNA reads a tensor and the DPU writes one. It stores the tensor as groups of channels, each group holding every width and height position for its channels ([encodings/tile-layouts.md](encodings/tile-layouts.md)) | Coined | NVDLA data cube in surface format, RKNN's NC1HWC2 |
| Cube-in | The consumer end of a cube join: a layer that reads its feature cube from another layer's output BO instead of packing a row-major input | Coined | |
| Cube-resident | Kept in the cube layout, in its BO, between two NPU operations, so that the next operation reads it as its input cube. No host de-tile or pack runs in between ([encodings/cross-op-chaining.md](encodings/cross-op-chaining.md)) | Coined | |
| `Dcal` | The negative log-likelihood, in nats, that an RK3576 W8A8 configuration adds against its control, summed over the `llama-perplexity` windows that are calibration forwards ([perf/rk3576-w8a8-llm.md](perf/rk3576-w8a8-llm.md)) | Coined | |
| DCOMP | Not expanded. The weight-decompression block inside the CNA, which expands compressed weights as the CNA loads them ([encodings/cna-dcomp-weight-decompression.md](encodings/cna-dcomp-weight-decompression.md)) | RK3588 TRM | |
| DDMA | Not expanded. The block of an NPU core that arbitrates its reads and writes to system memory (registers `0x8xxx`) | RK3588 TRM | NVDLA MCIF |
| De-tile | Converting a cube-layout output back to row-major order on the host | Coined | Unpack |
| Decode | The phase of LLM inference after prefill, which generates one token at a time: a matrix-vector product per weight (M=1, a GEMV) | Field | |
| Delta task | A task whose register program writes only the registers that differ from an earlier task's, and relies on the register groups for the rest ([encodings/regcmd-task-model.md](encodings/regcmd-task-model.md)) | Coined | |
| Dense writer | The RK3576 register program that writes an int8 program's accumulators as raw int32, through the whole output stage. It delivers every output channel ([chips/rk3576-regcmd.md](chips/rk3576-regcmd.md)) | Coined | |
| DPU | Data Processing Unit. The post-processing block of an NPU core: the BS, BN and EW stages, the LUT, the output converter and the WDMA (registers `0x4xxx`) | RK3588 TRM | NVDLA SDP |
| DPU_RDMA | The DPU's read DMA (registers `0x5xxx`), which reads the DPU's operands from memory through four channels: MRDMA, BRDMA, NRDMA and ERDMA | RK3588 TRM | NVDLA SDP_RDMA |
| Drain deadline | The time after `PC_DONE` that `rocket` waits for the DPU to finish writing before it signals a job's fence, set by `dpu_grace_us`. Erratum RK3576-E3 occurs when a drain outlasts it ([chips/rk3576.md](chips/rk3576.md)) | Coined | |
| `D_res` | The part of `Dcal` caused by the quantum of the int8 output that a call in a calibration forward returns, apart from clipping. It is read as the `Dcal` of a run with no row cut at the same bootstrap margin, `ROCKET_RK3576_BOOTMARGIN` ([perf/rk3576-w8a8-llm.md](perf/rk3576-w8a8-llm.md)) | Coined | |
| Encoder | The part of a model that turns its input into the hidden states its decoder or task head reads, as in Whisper or a ViT. The code that writes register programs is the regcmd encoder | Field | |
| ERDMA | E read DMA, with the E not expanded. The DPU_RDMA channel that reads the EW stage's second operand from memory | RK3588 TRM, NVDLA | |
| EW stage | Element-wise, as the TRM describes its operand. The third of the DPU's three affine stages, which can also combine each element with a second operand that ERDMA reads ([encodings/sdp-stage-precision.md](encodings/sdp-stage-precision.md)) | RK3588 TRM | NVDLA SDP Y |
| Expert stack | The weights of one MoE layer's routed experts, which llama.cpp holds as one tensor | Coined | |
| FA | Flash attention. Attention computed over blocks of keys with a running softmax, so that the whole matrix of scores is never held at once | Field | |
| Feature cube | An activation tensor in its cube layout | Coined | NVDLA feature data cube |
| FFN | Feed-forward network. The sub-block of a transformer layer that applies two or three matmuls, with an activation between them, to each token independently ([encodings/ffn-block.md](encodings/ffn-block.md)). Other sources call it the MLP, the multilayer perceptron | Field | |
| Flying mode | The input-source setting `flying_mode` = 1, in which a block reads its main input from memory instead of from the block before it. The DPU then reads through MRDMA instead of taking the convolution output, and the PPU reads memory instead of the DPU's output ([encodings/mrdma-trap.md](encodings/mrdma-trap.md)) | RK3588 TRM | NVDLA off-flying mode |
| GGUF | Not expanded. The model file format of ggml and llama.cpp, which holds a model's metadata and its weights, many of them in block-quantized types | ggml | |
| GQA | Grouped-query attention. An attention variant in which each group of query heads shares one key head and one value head, which shrinks the KV cache | Field | |
| Granule | 64 bytes of CBUF: the unit in which the RK3576 counts its CBUF allowances ([chips/rk3576-regcmd.md](chips/rk3576-regcmd.md)) | Coined | |
| Head chaining | Batching a worker fd's per-head attention-score matmuls into one submit, and its per-head value matmuls into a second, instead of one submit per head. `ROCKET_FA_CHAIN` enables it ([perf/attention-offload-crossover.md](perf/attention-offload-crossover.md)) | Coined | |
| Head fan-out | Splitting the attention heads of a multicore flash attention into head ranges over the worker fds, which the cores run in parallel ([perf/attention-offload-crossover.md](perf/attention-offload-crossover.md)) | Coined | |
| Head range | The contiguous range of attention heads that one worker fd computes in a multicore flash attention | Coined | |
| Ingest | Converting one MoE expert's GGUF-quantized weights into int8 codes and scales in resident NPU BOs. It runs the first time the router sends the expert a row ([perf/benchmarks.md](perf/benchmarks.md)) | Coined | |
| IOVA window | The IOVA range that one open fd's BOs are mapped into ([perf/iova-and-multicore.md](perf/iova-and-multicore.md)) | Coined | IOMMU address space |
| K-accumulation | Summing a matmul's partial results over successive K tiles. On chip, the DPU's EW stage adds each partial to the previous result ([encodings/k-accumulation.md](encodings/k-accumulation.md)) | Coined | Split-K reduction |
| K-group | The number of input channels (K) that the weight cube stores together, which depends on the datatype ([encodings/tile-layouts.md](encodings/tile-layouts.md)) | Coined | |
| K-split | Splitting a matmul's contraction (K) into chunks that run as separate tasks or matmuls, and summing their partial outputs. In the RK3576 W8A8 path of `ggml-rocket`, the host K-split quantizes and calibrates each chunk on its own ([perf/rk3576-w8a8-llm.md](perf/rk3576-w8a8-llm.md)) | Coined | Split-K |
| KACC | K-accumulation. The on-chip fp16 K-accumulation path, which the library option `ROCKET_KACC` enables | Coined | |
| Kick | One start of the PC by the kernel. Tasks that the hardware chains run in one kick, and tasks that the driver starts separately take one kick each | Coined | |
| KV cache | Key-value cache. The attention keys and values of the tokens processed so far, kept so that decode computes only each new token's | Field | |
| LE table | Linear/Exponent table. The DPU LUT's table whose entries can be spaced linearly or exponentially over its input range ([encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md)) | NVDLA | |
| LO table | Linear-only table. The DPU LUT's table whose entries are spaced linearly over its input range | NVDLA | |
| LUT | Lookup table. The DPU's table unit, which evaluates a nonlinear function per element ([encodings/dpu-lut-activation.md](encodings/dpu-lut-activation.md)) | RK3588 TRM, NVDLA | |
| Machine parameter | A per-SoC constant that software plans around. The CBUF size and bank count, the tile cap, the weight group sizes, the datatype menu and the core count are machine parameters ([chips/README.md](chips/README.md)) | Coined | |
| Margin index | The ratio of the clipping part of `Dcal` at a bootstrap margin of 1.00 to that at 2.25, both at a row cut to `M`/4. The clipping part is `Dcal` less `D_res` at the same margin, and the margin is `ROCKET_RK3576_BOOTMARGIN` ([perf/rk3576-w8a8-llm.md](perf/rk3576-w8a8-llm.md)). It is written `R` | Coined | |
| Micro-batch | llama.cpp's physical batch, `n_ubatch`: the number of prompt tokens that one graph evaluation processes, set with `-ub` | llama.cpp | ubatch |
| MLA | Multi-head latent attention. An attention variant that caches one compressed latent per token in place of each head's keys and values, which shrinks the KV cache | Field | |
| MoE | Mixture of experts. A model layer that sends each token to a few of its many expert sub-networks, chosen per token by a learned router | Field | |
| MRDMA | M read DMA, with the M not expanded. The DPU_RDMA channel that reads the DPU's main data from memory in flying mode. `mrdma_disable` turns it off ([encodings/mrdma-trap.md](encodings/mrdma-trap.md)) | RK3588 TRM, NVDLA | |
| MXFP4 | Microscaling FP4. A 4-bit floating-point format in which each block of 32 values shares one 8-bit power-of-two scale | OCP Microscaling Formats specification | |
| N-group | The number of output channels (N) that the weight cube stores together, which depends on the datatype ([encodings/tile-layouts.md](encodings/tile-layouts.md)) | Coined | |
| Narrow writer | The RK3576 register program that writes an int8 program's accumulators as raw int32, through a partial output stage. It delivers 8 output channels of every 32 ([chips/rk3576-regcmd.md](chips/rk3576-regcmd.md)) | Coined | |
| Native-quant path | The MoE expert path of `ggml-rocket`, `ROCKET_MOE_NATIVE`, that ingests each GGUF-quantized expert and runs it as resident int8 matmuls. The alternative dequantizes each expert to fp16 on every micro-batch | Coined | |
| NRDMA | N read DMA, with the N not expanded. The DPU_RDMA channel that reads the BN stage's operands from memory | RK3588 TRM, NVDLA | |
| Operating point | The conditions a performance figure holds under: the NPU clock, the core count, and whether the weights are resident. The kernel's OPP, an operating performance point, fixes only a clock and its voltage | Coined | Operating conditions |
| Output converter | The DPU's integer converter, which scales and shifts each accumulator into the output type, `(acc×SCALE)>>SHIFT`, through the `OUT_CVT_*` registers ([encodings/out-cvt-converter.md](encodings/out-cvt-converter.md)) | RK3588 TRM | NVDLA SDP output converter |
| packA | The host pass that scatters a matmul's activation (A) into the cube layout | Coined | GEMM operand packing |
| packB | The host pass that scatters a matmul's weight (B) into the cube layout | Coined | GEMM operand packing |
| Packed image | An input of four or fewer channels stored with each pixel's channels adjacent, as an image is. The CNA reads one straight from memory in its ARGB input mode, `ARGB_IN` ([chips/rk3576.md](chips/rk3576.md)) | Coined | NVDLA image-input convolution |
| Partial output stage | An RK3576 DPU output configuration for a four-byte element that changes only some of the whole output stage's words. Changing only the `out` field of `0x4010` is one. It triggers the wide-output hazard ([chips/rk3576.md](chips/rk3576.md)) | Coined | |
| PC | Not expanded. The block of an NPU core that fetches each task's register program from memory and writes each value into its block (registers `0x0xxx`). It then starts the blocks that the task's trailer names. The TRM's functional description calls it the register file fetch unit, and NVDLA has no counterpart | RK3588 TRM | |
| Per-axis | A quantization scale given per output channel rather than per tensor, as TFLite defines it | TFLite | Per-channel quantization |
| PPL | Perplexity. A language model's quality measure on a reference text, the exponential of its mean negative log-likelihood per token, where lower is better | Field | |
| PPU | Planar Processing Unit. The pooling block of an NPU core (registers `0x6xxx`), which has its own read DMA, PPU_RDMA (registers `0x7xxx`) | RK3588 TRM | NVDLA PDP |
| Pre-flight | The MoE residency check of `ggml-rocket`, run in `supports_op` before the first ingest. It reserves each expert stack's whole resident cost against the RAM and IOVA budgets. A stack that does not fit stays on the CPU ([TUNING.md](TUNING.md)) | Coined | |
| Prefill | The phase of LLM inference that processes the prompt as one batch: a matrix-matrix product per weight (a GEMM) | Field | |
| RDMA | Read DMA. A block's DMA channel for reading its operands from memory, as in DPU_RDMA and PPU_RDMA | NVDLA | |
| regcmd | Register command. One 64-bit word of a register program: a target block, a register offset and a 32-bit value ([encodings/regcmd-task-model.md](encodings/regcmd-task-model.md)) | `rknpu` driver | Command buffer entry |
| Regcmd encoder | The library code that writes register programs in one SoC's register encoding ([chips/porting-patterns.md](chips/porting-patterns.md)). Each SoC needs its own, because the encoding of the CNA, CORE and DPU geometry registers differs between SoCs | Coined | Command-stream generator |
| Register group | One of a block's two copies of its configuration registers. The block runs from one group while software programs the other | RK3588 TRM | NVDLA ping-pong registers |
| Register program | The sequence of regcmds for one task, ending in a trailer that starts the blocks ([encodings/regcmd-task-model.md](encodings/regcmd-task-model.md)) | Coined | Command buffer |
| Requant | Converting an integer accumulator to a narrower integer output with a scale and a shift | Field | Requantization |
| Resident weights | Weights packed into the cube layout once, at model load, and kept in BOs, rather than packed on every call | Coined | |
| RKNPU | Rockchip's neural processing unit, derived from NVIDIA's NVDLA, which the RK3588 TRM documents as RKNN. Its datapath semantics are common to every SoC that integrates it, and its register encoding can differ between revisions ([chips/porting-patterns.md](chips/porting-patterns.md)) | `rknpu` driver | NPU |
| `rknpu` | Rockchip's vendor driver for the RKNPU, in Rockchip's BSP kernel | `rknpu` driver | |
| `rocket` | The mainline Linux DRM accel driver for the RKNPU, `drivers/accel/rocket`. It allocates buffers and submits register programs through the ioctls `CREATE_BO`, `SUBMIT`, `PREP_BO` and `FINI_BO` | Linux kernel | |
| Row cut | Running the bootstrap's first pass over only the first rows of a call in a calibration forward, as many as `ROCKET_RK3576_CALROWS` sets. A cut is written as the rows kept, as in cut 512, or as a fraction of `M`, as in cut `M`/4 ([perf/rk3576-w8a8-llm.md](perf/rk3576-w8a8-llm.md)) | Coined | |
| Row window | The range of input rows that one RK3576 convolution task reads when a convolution is split by input rows to fit the CBUF allowance. The task writes the output rows that the range supports ([chips/rk3576-regcmd.md](chips/rk3576-regcmd.md)) | Coined | |
| Row-major form | A matmul program that reads its activation and writes its output as plain rows, without packing either into the cube layout. It programs the CNA's and the DPU's strides instead ([matmul-as-conv.md](matmul-as-conv.md)) | Coined | |
| SDMA | Not expanded. The block of an NPU core that connects it to on-chip SRAM (registers `0x9xxx`) | RK3588 TRM | NVDLA SRAMIF |
| Sentinel stamp | A known byte pattern written into an output BO, inside a bracket, before a submit. An element the NPU never wrote then differs from one it wrote as zero | Coined | |
| Shift word | The 32-bit per-task operand that the RK3576's DPU reads through `BS_BASE_ADDR1` (`0x5024`). It contains the BS stage's right shift for a non-negative product in bits [5:0], and for a negative product in bits [13:8] ([chips/rk3576-regcmd.md](chips/rk3576-regcmd.md)) | Coined | |
| Streaming path | The matmul path that packs its weights into the cube layout on every call, the alternative to resident weights | Coined | |
| Submit | One `SUBMIT` ioctl to `rocket`: a job of one or more tasks | Coined | Job |
| Super-group | A group of 32 output channels in the wide writer's output, which the DPU writes as one stream of `4·A` atoms. `A` is the output's pixel count ([chips/rk3576-regcmd.md](chips/rk3576-regcmd.md)) | Coined | |
| Surface | One plane of a cube layout: every width and height position of one atom-wide group of channels. A surface stride is the distance between consecutive surfaces, and the notes do not use the word for an output tensor | NVDLA | |
| Task | One register program with its descriptor, `{regcmd address, regcmd count}`: the unit the PC executes | RK3588 TRM | NVDLA hardware layer |
| TDT | Token-and-duration transducer. A speech-model decoder that predicts each token together with the number of audio frames it spans | Field | |
| Test point | One combination of the swept parameters in a measurement | Coined | Design point |
| Tile | The part of a matmul that one task computes: `Mt × Nt` outputs over `Kt` of the contraction | Field | |
| Transform-bound | Limited by the host's layout conversions, the pack into the cube layout and the de-tile out of it, rather than by NPU execution ([encodings/cross-op-chaining.md](encodings/cross-op-chaining.md)) | Coined | |
| TTFT | Time to first token. The time from a prompt's submission to the first generated token, which prefill sets | Field | |
| ViT | Vision transformer. An image model that splits an image into patches and runs them through transformer layers as tokens | Field | |
| W8A8 | 8-bit weights, 8-bit activations. A quantization scheme with int8 weights and int8 activations | Field | |
| WDMA | Write DMA. The DPU's DMA channel that writes its output to memory | RK3588 TRM, NVDLA | NVDLA SDP WDMA |
| Weight cube | A weight tensor in its cube layout ([encodings/tile-layouts.md](encodings/tile-layouts.md)) | Coined | NVDLA weight data cube |
| Weight slice | The part of the weight cube that a task keeps in the CBUF while it runs ([chips/rk3576-regcmd.md](chips/rk3576-regcmd.md)) | Coined | |
| WER | Word error rate. A transcript's word-level edit distance from a reference transcript, divided by the reference's word count | Field | |
| Whole output stage | The complete set of nine RK3576 DPU words that configure the output for a four-byte element. A program that sets all of them writes int32 or fp32 without the wide-output hazard ([chips/rk3576.md](chips/rk3576.md)) | Coined | |
| Wide writer | The RK3576 register program that writes an int8 program's accumulators as raw int32, through a partial output stage. It delivers 8 output channels of every 16 ([chips/rk3576-regcmd.md](chips/rk3576-regcmd.md)) | Coined | |
| Wide-output hazard | RK3576 erratum RK3576-E2. A program that writes a DPU output element wider than one byte through a partial output stage leaves the next submit unwritten. The next submit can be of any kind and from any process ([chips/rk3576.md](chips/rk3576.md)) | Coined | |
| Worker fd | One of the several fds that a multicore matmul opens on the NPU's device node, to spread its tiles across the cores ([perf/iova-and-multicore.md](perf/iova-and-multicore.md)) | Coined | |
| Write guard | The RK3576 check in `librocketnpu` that every task of a submit wrote its output, and that the job completed before the backstop. A submit that fails it is redone after a power-domain cycle ([chips/rk3576.md](chips/rk3576.md)) | Coined | |

## Notation

| Symbol | Meaning |
|---|---|
| M, K, N | A matmul's dimensions, `C[M×N] = A[M×K] × B[K×N]`. Run as a 1x1 convolution, M is the spatial height, K the input channels and N the output channels |
| Mt, Kt, Nt | A tile's extents in M, K and N |
| IC, OC | A convolution's input and output channel counts |
| IH, IW, OH, OW | A convolution's input height and width, and its output height and width |
