# RT-core batched BP for beam-search decoding: project plan

Plan for a time-boxed project of 3 to 5 days. Every day ends on a **tagged, working, benchmarked version**, so if you run out of time you hand in the last tag.

---

## 0. What the research changed (read this first)

| Finding | Consequence for the plan |
|---|---|
| **Your laptop has only an Intel Iris Xe GPU.** WSL shows no `nvidia-smi`, no CUDA and no RT cores. It does have g++ 11.4, CMake 3.22, 12 cores and Python 3.10. | Do all CPU work locally (reference decoder, data export, golden test set). Rent an NVIDIA RTX-class GPU for everything else. |
| **A100, H100 and B200/GB200 have no RT cores.** The published "SOTA GPU decoder" numbers were measured on those chips. | Rent an **RTX 4090, L40S, L4, A10/A10G or RTX 6000 Ada**. Then run the SOTA decoder (CUDA-Q QEC) yourself on the same card, so the comparison is fair. |
| OptiX's runtime library ships with the **driver**, as `libnvoptix.so.1` plus `libnvidia-rtcore`. Containers often don't mount it, which gives `OPTIX_ERROR_LIBRARY_NOT_FOUND`. | Day 0 includes a **15-minute OptiX smoke test** on the rented machine. If it fails in a container, switch to a VM-style provider (AWS g6/g6e, GCP g2, Lambda A10/A6000) instead of debugging. |
| You can't reach RT cores from plain CUDA. You need OptiX, Vulkan or DXR. OptiX programs also have **no shared memory and no `__syncthreads`**. | Each BP half-iteration has to be its own `optixLaunch`, so 20 iterations means about 40 launches per round. At batch size 1, launch overhead will dominate the RT path. This is a structural fact to measure and report, not a bug to fight. |
| **IonQ's BeamSearchDecoder is single-threaded C++ with a Cython wrapper.** It uses plain min-sum (no scaling) and a flooding schedule. A beam candidate is a set of fixed `(bit, value)` pairs. Each child runs masked BP for `iters_per_round` iterations, warm-started from its parent's messages. It branches on `argmin |Σ_iter LLR|` over unfixed bits with column degree > 2. The score is `Σ_unfixed |LLR_sum| / iters_run`, and the beam keeps the top `beam_width` scores. It returns the first converged child. There is no OSD, no GPU code and no batching. Circuits are included only for BB [[144,12,12]], BB [[90,8,10]] and HGP [[450,32,8]]. | This is your **oracle**. On a fixed syndrome set, aim for the GPU decoders to produce the *same corrections* as this code, not just a similar logical error rate (LER). |
| **Paper numbers (arXiv 2512.07057)** are for BB144, 12 rounds, on one M3 Pro core, in ms per full decode. At p=1e-3: BP-OSD averages 10.59 ms with a p99.9 of 289 ms. beam8 averages 2.32 ms (p99.9 11.0 ms). beam32 averages 3.84 ms (p99.9 14.2 ms). beam64 averages 6.70 ms (p99.9 17.7 ms). The paper's "<1 ms" claim is **per syndrome round**, meaning the total divided by 12: beam32 at p=5e-4 gives 0.27 ms average and 0.94 ms p99.9. For LER, beam32, beam64 and beam64 with 32 results are 5.6×, 7.0× and 17× better than BP-OSD. | Report latency both per decode and per round, so you're comparable with the paper. |
| **SOTA GPU reference.** NVIDIA CUDA-Q QEC `nv-qldpc-decoder` (`pip install cudaq-qec`) offers batched BP+OSD and Relay-BP. One independent benchmark (arXiv 2605.30765) ran BP30+OSD-0 on an A100 at batch 256 and p=1e-3. It reports **[[72]] 8.2 µs/round, [[144]] 42.5 µs/round and [[288]] 239.5 µs/round** (mean, amortized). Vendor docs give only relative numbers. | Use **42.5 µs/round for BB144** as the throughput target to beat, re-measured on your own GPU. |
| **RT cores outside graphics (review, arXiv 2603.28771).** Nearest-neighbour search and heuristic pruning gain the most, up to 200× for kNN. Many short rays beat few long ones. Results range from several times *slower* to much faster. Only BVH traversal and triangle tests run in hardware; any-hit and intersection programs run on the SMs. | Expect RT-traversal BP on a **static** graph to lose to a tuned CSR kernel. Plan for RT to earn its keep on **dynamic** structure: pruning, culling and ray dispatch (Task D). A well-measured negative result for Task C is still a valid Task C. |
| **The problem is tiny for a GPU.** BB144 with 12 rounds is about 1k detectors × 8.8k fault mechanisms, so roughly 10⁴–10⁵ Tanner edges (measure exact `m, n, nnz` on Day 0). The worst case for beam8 is `30 + 10·16·20 = 3230` BP iterations per shot. | Latency is governed by launches and synchronization, not FLOPs. The wins come from keeping a whole child's BP inside one kernel, from CUDA Graphs, and from batching shots for throughput. |

### Working hypotheses (put these in the report)
- **H1.** On a static Tanner graph, RT-traversal BP is slower than CSR CUDA BP, because of BVH traversal overhead, any-hit calls that bounce to the SMs, and two launches per iteration.
- **H2.** RT becomes competitive when the graph to traverse changes per shot and per round, as with active-set pruning, converged-candidate culling or dynamic ray dispatch. In that setting a BVH rebuild or refit plus spatial culling replaces irregular index compaction.
- **H3.** At batch size 1 the GPU is mostly idle: beam8 has only 16 children per round on more than 100 SMs. That makes **wider beams almost free in latency**, so you get better LER at the same latency, which a single CPU core can't offer.

---

## 1. Fixed decisions (so you don't spend time re-deciding)

- **Stack.** Python handles orchestration, using stim and sinter for sampling and LER. CUDA kernels go in **CuPy `RawModule`** (no build system). OptiX is driven from Python with **`otk-pyoptix`** (NVIDIA) or **`pyoptix-contrib`** (the OptiX 9.1 fork), with device code compiled through NVRTC. Headers come from `github.com/NVIDIA/optix-dev`.
  - *Plan B, decided after at most 2 hours of install trouble:* write a standalone C++ OptiX binary, starting from the `optixHello`/`optixTriangle` pattern. It reads `.npy`/`.bin` inputs, times internally and writes results. Python just shells out to it.
- **One decoder interface.** Every decoder implements `decode_batch(syndromes: uint8[B, m]) -> (corrections: uint8[B, n], stats)`. That covers the IonQ CPU decoder, the NumPy reference, CUDA v1 and v2, the RT variants and CUDA-Q QEC. A single `bench.py` then compares any subset, which is what makes falling back painless.
- **Primary benchmark.** BB [[144,12,12]], 12 rounds, memory_Z (plus memory_X for LER), p ∈ {5e-4, 1e-3}, configurations beam8_230 and beam32_340. Everything else is a Task E extension.
- **Repository layout**
  ```
  rtbeam/
    data/        exported H/priors/observables + golden syndromes & IonQ outputs (.npz)
    ref/         numpy reference beam search (spec for the kernels)
    cuda/        bp_children.cu, beam_select.cu
    optix/       rt_bp.cu (raygen/anyhit programs), host wrapper
    rtbeam/      python package: decoders behind the common interface
    bench/       bench.py (latency/throughput), ler.py (sinter), plots.py
    results/v0.1, v0.2, ...   (one folder per tag, regenerated by one command)
    REPORT.md    (written incrementally, one section per day)
  ```
- **Fallback discipline**
  - `main` always runs `python bench/bench.py --quick` successfully.
  - Do experimental work on a branch and merge only when the bench passes.
  - `git tag vX.Y` at the end of every day.
  - Spend the last 45 minutes of each day on results and REPORT.md, never on code.

---

## 2. Day-by-day timeline

Assume about 9 hours of focused work per day. Time boxes are hard limits: when one expires, take the **"If behind"** branch.

### Day 0: prep on your laptop (about 3 hours, no GPU needed)
If you can't do this beforehand, it becomes Day 1 morning and you should switch to the 3-day cut.

1. Clone `ionq-publications/BeamSearchDecoder`, run `pip install -r requirements.txt`, then `cd decoder && python3 setup.py build_ext --inplace`. Run code from the repo root.
2. `export_dem.py`: convert each `.stim` file to a DEM with `stimbposd.dem_to_matrices`, and save H (CSR + CSC), priors and the observables matrix to `.npz`. **Record `m, n, nnz`, the max row and column degree, and the number of columns with degree ≤ 2.**
3. Golden set: about 2k seeded shots per (p, config). For each shot store the syndrome and IonQ's correction, `converged`, the number of rounds used and the per-shot time. Also record the IonQ single-core latency (mean, p99.9) and the all-core throughput through sinter workers. These are your CPU baselines.
4. Pick and book a GPU (see section 4). Run the **smoke test** on it:
   - `nvidia-smi`
   - `ldconfig -p | grep -E 'nvoptix|rtcore'`
   - the pyoptix triangle example
   - `pip install cupy-cuda12x cudaq-qec`, then a 10-line `nv-qldpc-decoder` decode on BB144

   **If OptiX fails in a container, switch providers right away.**

### Day 1: Task A, baseline GPU beam search → tag **v0.1**
| Block | Work | Done when |
|---|---|---|
| 2 h | **NumPy batched reference** (`ref/`): vectorize over the children of a round, and match IonQ semantics exactly: min-sum without scaling, `≤ 0 → 1` hard decision, degree > 2 branching filter, score normalization, top-k, dedup rule, syndrome XOR for bits fixed to 1, warm-started messages, and the fallback to the initial BP decision. | ≥ 99% identical corrections on the golden set. The rest should be float tie-breaks; inspect a few. |
| 4 h | **CUDA v1** (CuPy RawModule). `bp_children`: **one CTA per (shot, child)**, with all `iters_per_round` iterations inside the kernel and `__syncthreads` between the check and variable sweeps. Messages are fp32 in global memory. It accumulates LLR_sum, checks convergence, and does a block-level argmin for the next branch bit and the score. `beam_select`: one CTA per shot, top-`beam_width`, done flags, new beam with parent-state copies. The host loops over rounds and exits early when all shots are done. **Return the converged child the CPU would have reached first** (parent order by descending score, value 0 before 1); that keeps the output bit-exact. | Matches the NumPy reference and IonQ on the golden set. |
| 2 h | `bench.py`: batch=1 latency (CUDA events and end-to-end wall time, mean/p50/p99/p99.9 over ≥ 10k shots) and throughput (shots/s for batch ∈ {1, 16, 256, 4096}). `ler.py`: sinter wrapper, X+Z memory, the same per-round LER formula as IonQ's `one_point_simulation`. | Table: IonQ CPU vs CUDA v1. |
| 45 min | REPORT §A, plots, `git tag v0.1` | |

**If behind:** use a pure-CuPy vectorized version (segment reductions over CSR) as the "GPU baseline" and tag it. That still satisfies Task A.

### Day 2: Task B, initial RT mapping → tag **v0.2**
**Geometry, one triangle per Tanner edge.** Variable `v` maps to `x = v` and check `c` maps to `y = c`. Edge `(c, v)` becomes a small triangle centred at `(v, c, 0)` that lies in the diagonal plane `x + y = v + c`, with extent < 0.5. Because the plane is diagonal, the same triangle is hit both by the **row ray** of check `c` (origin `(xmin(c) − ½, c, 0)`, direction `+x`) and by the **column ray** of variable `v` (direction `+y`). One GAS holds both passes, and the intersection test runs in hardware.

**Compact min-sum on rays.** Both passes are reductions that don't depend on hit order, which suits any-hit enumeration (call `optixIgnoreIntersection()` to keep traversing).
- **Check pass (row rays):** any-hit reads edge `e = optixGetPrimitiveIndex()`, skips masked bits, computes `v2c = posterior[v] − c2v_old(e)`, and folds it into the payload (`min1, min2, argmin, sign parity`). Ray-gen then writes the compact check state and the v2c sign bits.
- **Variable pass (column rays):** any-hit rebuilds `c2v(e)` from the compact state and sums it. Ray-gen writes the posterior, LLR_sum and hard decision.
- **Syndrome check:** row rays accumulate the parity of the hard decisions.
- **Batching:** the launch grid is `(rays × children × shots)` over the **same GAS**. The geometry is shared, and only the state indices change.

| Block | Work | Done when |
|---|---|---|
| 2 h | OptiX plumbing: build the GAS from H, and run row and column enumeration queries. | Each check and variable gets exactly its CSR or CSC neighbour set. |
| 3 h | RT plain BP (no beam), batched over shots. | LLRs match the CUDA BP after N iterations (fp32, same operation order where possible). |
| 2 h | Plug RT BP into the beam: the RT passes replace the inside of `bp_children`, and `beam_select` stays CUDA. | Golden set matches. |
| 1 h + 45 min | First RT numbers, REPORT §B, `git tag v0.2` | |

**If behind:**
- If OptiX install or plumbing passes 2 hours, switch to Plan B (the C++ binary).
- If both fail, implement the same mapping on a **software BVH**, either NVIDIA Warp `wp.Mesh` ray queries or your own LBVH in CUDA. Label it clearly as "BVH mapping, no RT cores". That is weaker but still demonstrates the mapping.
- Plain RT BP without the beam is an acceptable v0.2.

### Day 3: Task C, batched BP optimization and evaluation → tag **v0.3 = PASS**
| Block | Work |
|---|---|
| 3 h | **CUDA v2.** This is your fastest decoder and makes the comparison fair. Keep the compact min-sum state per child (check min1/min2/argmin/parity, v2c sign bits, posterior) in **shared memory**; check whether it fits, and use fp16 for some arrays if needed. Use a warp per check for row sweeps. Use **CUDA Graph** capture of the round loop and compaction of converged shots and children. |
| 3 h | **RT ablations**, one row each in a table: (i) graph reordering (reverse Cuthill–McKee on rows and columns) so rays get shorter and the BVH tighter, (ii) splitting rows into k short rays ("many short rays"), (iii) build flags (`PREFER_FAST_TRACE`, compaction), (iv) `DISABLE_CLOSESTHIT`, `REQUIRE_SINGLE_ANYHIT_CALL`, minimal payload, (v) launch shape (children of one shot adjacent), (vi) CUDA-graph capture of the launch sequence (check whether your OptiX version supports stream capture). Log rays per iteration and hits per ray. |
| 2 h | **Evaluation run** at batch=1 latency and batch-sweep throughput, with LER parity for every decoder: IonQ CPU (1 core and all cores), `ldpc`/`stimbposd` BP-OSD, CUDA v1, CUDA v2, RT best, and `cudaq-qec` BP+OSD and Relay-BP on the same GPU. |
| 1 h | REPORT §C states plainly where RT wins or loses, and why (H1). `git tag v0.3`. **This is a complete Pass submission.** |

**If behind:** drop ablations (v) and (vi), and drop Relay-BP. Keep CUDA v2 and the evaluation table; those are what Task C is graded on.

### Day 4: Task D, advanced RT pruning and ray dispatch → tag **v0.4**
Ordered by certainty of payoff. Do them in order and tag after each one that works.

1. **Dynamic ray dispatch (1.5 h, safe win).** Each round, compact the list of live `(shot, child)` pairs and of checks or variables that are still active. Launch rays only for those, and drop converged children and shots. Measure throughput when shots converge at mixed rates.
2. **Wide beam at fixed latency (1 h, safe win, tests H3).** At batch=1, compare beam8, beam32 and beam64 latency against LER. If beam64 costs about the same as beam8 on the GPU, you get about 7× better LER at the same latency.
3. **Batched dynamic BVH pruning (4 h, the RT research piece, tests H2).** For each shot, keep only the **active edges**, meaning those within k hops of unsatisfied checks (tune k between 2 and 4) and optionally excluding edges with saturated, unchanging messages.
   - Build **one GAS per round for the whole batch**: shot `s`'s active triangles go into z-slab `z = 2s`, and its rays stay in that slab. That's one build per batch, not per shot.
   - Between rounds, shrink the GAS. Either rebuild it, or move dropped triangles out of range and **refit**, which is cheaper.
   - First validate the pruning algorithm in CUDA with an active edge list, and confirm the LER doesn't degrade. Then compare CUDA active lists against RT dynamic BVH to isolate what RT adds.
4. *Stretch, only if 1–3 are done:* an RTNN-style nearest-neighbour cache of past syndromes, as defect point clouds in detector coordinates, used to warm-start priors or seed beam candidates.

Spend 1 h on REPORT §D and `git tag v0.4`.

**If behind:** items 1 and 2 alone are a defensible Task D ("hardware-aware ray dispatch and beam scaling"). Write item 3 up as a design with the CUDA-side pruning results.

### Day 5: Task E, scalability and hardware efficiency → tag **v1.0**
- **Codes**
  - BB [[90,8,10]] and HGP [[450,32,8]] come from the IonQ repo.
  - Generate BB [[72,12,6]] and [[288,12,18]] with `build_circuit.py` from `gongaa/SlidingWindowDecoder`, which follows the Bravyi et al. circuit. It is a different construction from IonQ's, so also regenerate BB144 with it once to check consistency.
- **Sweep.** Codes × p ∈ {5e-4, 1e-3, 2e-3} × batch ∈ {1, 64, 1024, 8192}.
  - **Metrics:** mean/p99/p99.9 latency per decode and per round, shots/s, LER, and GPU efficiency (edge updates/s, rays/s, achieved occupancy, an Nsight Systems timeline, and power from `nvidia-smi` for energy per shot).
- **Report "sub-millisecond" three ways, honestly:**
  - (a) per-round p99.9 < 1 ms, which the CPU already achieves
  - (b) amortized time per shot < 1 ms at large batch
  - (c) a full 12-round decode with p99.9 < 1 ms at batch=1. This is the hard one; say whether you hit it.
- Write the final REPORT and `git tag v1.0`.

---

## 3. Shorter or longer versions

| Days available | Schedule | Ends at |
|---|---|---|
| **3** | Day 0 prep happens on Day 1 morning. Day 1 is A, Day 2 is B, Day 3 is C. | v0.3, **Pass** (A, B, C) |
| **4** | Days 1–3 as above. Day 4 morning: D items 1 and 2. Day 4 afternoon: an E-lite sweep (BB72, BB144, BB288 at p=1e-3, batch {1, 1024}). | v1.0-lite, a credible attempt at full points |
| **5** | Exactly as in section 2. | v1.0 |
| **More** | Give D item 3 more time, and add Relay-BP-style ensembles inside the beam. | |

Mapping to the grading criteria: **A** is v0.1, **B** is v0.2, **C** is v0.3 (Pass), **D** is v0.4, and **E** is v1.0 (full points).

---

## 4. Hardware and setup checklist

- **GPU:** an RTX 4090 or L40S (Ada, 3rd-gen RT cores) is the best value. An L4 or A10G works on AWS or GCP VMs, where OptiX in a VM is straightforward. **Avoid A100 and H100: they have no RT cores.** Budget roughly 25–40 GPU-hours; check current marketplace prices.
- **Driver:** recent enough for your OptiX version; see the minimum driver in the OptiX release notes.
- **Python packages:** `cupy-cuda12x stim sinter ldpc stimbposd cudaq-qec pybind11 cmake`, plus the IonQ requirements.
- **OptiX:** headers from `NVIDIA/optix-dev`. Python bindings from `otk-pyoptix`, or `pyoptix-contrib` for OptiX 9.1; both build from source and need the headers present.
- **Remote workflow:** develop over VS Code Remote-SSH, or run Claude Code directly on the GPU box. Keep the laptop for the CPU oracle and writing.
- **Licence:** the IonQ repo is **CC BY-NC-SA 4.0**. That's fine for coursework; attribute it, and note that copied code inherits ShareAlike.

## 5. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| OptiX not usable on the rented box | Medium | Day 0 smoke test; VM provider; Plan B C++ binary; software-BVH fallback |
| GPU output diverges from IonQ (ties, fp32) | Medium | NumPy reference first; deterministic "first converged child" rule; compare decisions, not just LER |
| RT slower than CUDA everywhere | **High** | Expected (H1). Make it a measured result; put RT effort into dynamic structure (D item 3) |
| Launch overhead swamps batch=1 latency | High | In-kernel iterations for CUDA; CUDA Graphs; report latency and throughput regimes separately |
| `cudaq-qec` won't install or run on the consumer GPU | Low–medium | Fall back to the published A100 numbers, clearly labelled as different hardware |
| Too few shots for meaningful LER | Medium | Fix the shot budget early; use the GPU decoder itself for LER runs, since it's the fast one |

## 6. Prior art to cite and position against

- IonQ beam search decoder: [paper](https://arxiv.org/abs/2512.07057), [code](https://github.com/ionq-publications/BeamSearchDecoder)
- RT cores for general-purpose compute, a review: [arXiv 2603.28771](https://arxiv.org/abs/2603.28771)
- RTNN, neighbour search on RT cores (PPoPP'22): [arXiv 2201.01366](https://arxiv.org/abs/2201.01366)
- RTXRMQ, range-minimum queries on RT cores: [arXiv 2306.03282](https://arxiv.org/abs/2306.03282)
- RTSpMSpM, sparse matrix multiplication on RT hardware (ISCA'25): [summary](https://danglingpointers.substack.com/p/rtspmspm-harnessing-ray-tracing-for)
- CUDA-Q QEC GPU decoders: [docs](https://nvidia.github.io/cudaq-qec/), [performance studies](https://nvidia.github.io/cudaq-qec/performance/index.html)
- Independent GPU-vs-CPU BP+OSD benchmark: [arXiv 2605.30765](https://arxiv.org/abs/2605.30765)
- BB circuit generator and BP-GDG: [gongaa/SlidingWindowDecoder](https://github.com/gongaa/SlidingWindowDecoder), [arXiv 2403.18901](https://arxiv.org/abs/2403.18901)
- OptiX: [headers](https://github.com/NVIDIA/optix-dev), [otk-pyoptix](https://github.com/nvidia/otk-pyoptix), [pyoptix-contrib](https://pypi.org/project/pyoptix-contrib/), container caveat: [NVIDIA forum](https://forums.developer.nvidia.com/t/optix-7-5-8-0-fails-inside-docker-but-works-on-host/280500)

*Not yet verified, so check these before relying on them: exact DEM sizes for the IonQ circuits; the 2605.30765 numbers, which came from a model-extracted summary of the HTML; whether `cudaq-qec` supports consumer GPUs; whether `optixLaunch` can be captured into CUDA graphs on your OptiX version.*
