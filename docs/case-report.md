# State (Arc Institute) on Tahoe-100M — a frozen-judge case

One training step of the [State](https://github.com/arcinstitute/state) transition model (`tahoe_best`,
209.1 M parameters) on a slice of [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M), judged on
one A800. This is the write-up of the case: what it measures, what it found, and — in section 4 — what the
first version of it got wrong.

---

## 1. The workload

| | |
|---|---|
| Upstream | `github.com/arcinstitute/state` @ `9bbfe78` |
| Config | `tahoe_best` (`cell_set_len=512`) |
| Step | batch 8 cell sentences x 512 cells = 4096 cell rows |
| Parameters | 209.1 M |
| Hardware | 1 x A800 80 GB PCIe, `world_size=1` |

The slice: **13 plates, 50 cell lines, 380 drugs, 1243 samples — 399,990 cells x 47,942 genes.** Built with a
per-shard quota, because Tahoe-100M's shards are ordered by plate and a head-of-file slice contains a single
plate. All-zero genes (24.5 % of the panel) are dropped.

## 2. The judge

```
python bench/bench.py check --final --tree <candidate> --out <dir>
```

| gate | question |
|---|---|
| 0 structure | did the tree change something that makes the measurement meaningless? |
| 1 benefit | is the candidate faster than **two live reference runs** in the same session? |
| 2 record | are the recorded tensors of the eager path bit-identical to the reference? |
| **3 batches** | does the candidate's own dataloader produce the reference's batches, element for element? |
| **4 trajectory** | does the candidate's real training trajectory stay within the reference's? |

Gate 2 is a *record*, not the verdict: it drives a standalone adapter, so it covers the eager path only. Gates 3
and 4 run the timed configuration and carry the verdict jointly. Gates 3 and 4 exist because of section 4.

**Gate 3 (batches, bitwise).** Each tree's own dataloader is run with the same seed and sampler, 10 batches,
and every leaf tensor is compared. This is the check the recipe's "the pipeline may change as long as the
batches are identical" always needed: without it, `fast_fetch`, the CSR batch and the on-device densify were
never exercised by any gate.

**Gate 4 (trajectory, statistical).** `state tx train` at three seeds for 12 steps, recording a fixed sample of
the optimizer's parameters after each step. The timed path is not bitwise reproducible — flash attention's
backward accumulates with atomics — so the question is whether the candidate stays inside the reference's own
measured spread. The bound is frozen from **ten repeats of the reference per seed**, before any candidate is
recorded.

Each comparison is a z score: the deviation of one recorded statistic from the reference mean, divided by the
reference's own standard deviation for that statistic at that step. Compared are the L2 norm and the largest
magnitude of each of 24 sampled parameter tensors, plus every logged loss that is an output rather than a
diagnostic.

The threshold is not 3. A per-comparison 3 sigma applied to the worst of ~1500 comparisons is not a 3 sigma
event, and it rejects the reference itself: leave-one-out over the 30 reference recordings puts the reference's
own worst z between 2.19 and 6.10, and a three-seed check of the reference passes a 3.0 bound only **3 %** of
the time. The frozen threshold is the smallest value at which the reference passes 95 % of its own checks:
**6.11**. The reference is the gate's positive control and is reported next to the `lr x 1.02` arm.

The maximum of many comparisons is a tail statistic, and its threshold is estimated from few draws. The gate
therefore reports two statistics that do not take a maximum — the median z and the share of comparisons beyond
3 sigma — with a bound set at the worst value the reference itself produced (1.8 %). The verdict requires both
checks; on every arm below they agree.

## 3. The judge was tested against known-wrong candidates

| candidate | expected | observed |
|---|---|---|
| unmodified reference | no benefit | `UNAVAILABLE` / `no_benefit`, 0.9384x |
| control arm `lr x 1.02` | gate 1 (1.03x is not a speedup) | `UNAVAILABLE` / `no_benefit` |
| tampered frozen file | gate 0 | `ERROR` |

## 4. What the first version of this judge got wrong

The first version claimed tier **N0** — 794 of 794 recorded tensors bit-identical — and that claim was wrong in
a way worth recording, because each defect is a general way a frozen judge can be satisfied by something it
never looked at.

1. **It judged a different configuration from the one it timed.** The record always ran with deterministic
   algorithms on; the optimized tree's compiled blocks and CUDA graph ran only when deterministic mode was
   *off*. So gate 2 recorded an eager shadow of the step that gate 1 timed. Fixed by adding gates 3 and 4 on
   the timed path.
2. **It never checked the data pipeline.** The adapter replayed a frozen fixture and densified it itself, so the
   three commits that took the step from 0.368 to 0.212 s were outside every gate. Fixed by gate 3.
3. **It timed one precision and recorded another.** The frozen launch path sets
   `float32_matmul_precision("medium")`, so the timed step runs TF32; the record ran at torch's default,
   `highest`. The same 4096x2048x47942 GEMM measures 66.9 TFLOPS under `medium` and 18.4 TFLOPS under
   `highest` — a 3.6x difference in what the two paths were doing.
4. **Flash attention's backward is not deterministic** (479 differing tensors in the first reference record).
   Freezing the reference required a deterministic SDPA stand-in.
5. **The adapter measured the pip-installed package, not the candidate tree.** A control arm falsely passed.
6. **The adapter built its own optimizer** with `weight_decay=5e-4`, while upstream uses `Adam(lr)`. Every
   candidate was judged against a different optimizer.
7. **The reference was recorded on the host but judged in a sandbox** with a different interpreter prefix. The
   RNG stream shifted deterministically while the tensors stayed identical — a defect only visible once the two
   environments were compared.
8. **The sandbox hid `$HOME`, where the interpreter lived.** The first optimization run could not find Python.
9. **The bench could not run the harness's own check.** `judge.sh` calls `check --final`; the bench rejected
   the flag. Added, with a `task/final-evaluation` marker the candidate cannot create.
10. **The loader dropped the statistics the judge was reading.** When the judged statistic moved from a tensor
   *sum* to the L2 norm and largest magnitude, `components()` was updated and `load()` was not: the loader still
   carried only `sums` and the logged losses, while `components()` looked for `norms` and `absmax`, found none
   and skipped them. Gate 4 silently compared two loss scalars per step and **none of the 24 parameter tensors**.
   It made every verdict look milder and every calibration look tighter than they were: the reference's own
   leave-one-out worst z appeared to be 0.94–3.50 with a threshold of 3.51, where the full statistic gives
   2.19–6.10 and 6.11. The defect was found by a reviewer who could not reproduce the published calibration.
   Fixed, the frozen bound regenerated, and the judge now reports what it actually compared.

## 5. Results

Two goal runs on the same frozen judge. They are reported separately because they answer different questions.

| run | start | candidate | harness verdict |
|---|---|---|---|
| 1 — cold, unmodified tree | `9bbfe78` | 0.1883 s/iter | **1.885x** (2.13x against the reference tree's own 0.396) |
| 2 — warm, from run 1's tree | run 1 | 0.1789 s/iter | **2.02–2.08x** |

Round 1 is the comparable number: it started from the untouched tree with nothing but the task text. Round 2
started from round 1's tree, in a workspace that held round 1's work log, a hotspot map written by the
formulator, and a goal line naming 1.885x.

Three denominators appear and are not interchangeable: the **live reference** timed in the same session (what
gate 1 divides by; 0.355–0.386 s/iteration, 11 % spread on this shared host), the **frozen** 0.3713
(`task/timing_ref.json`), and the **unmodified tree** measured at the time (0.3956).

### What the trajectory gate says about each tree

The gate records `state tx train` at three seeds for 12 steps and compares each run against the bound of
section 2 — z per (step, component) against ten repeats of the reference per seed, threshold 6.11, written
before any candidate was recorded.

| tree | speedup | worst z | steps outside | median z | beyond 3σ | verdict |
|---|---|---|---|---|---|---|
| **data path only, no compile (`b438d76`)** | **1.675x** | **3.00** | **0 / 30** | **0.08** | **0.07 %** | **equivalent** |
| data path + `torch.compile` (`4b0d91d`) | 1.885–1.934x | 16.05 | 27 / 30 | 0.39 | 15.4 % | different |
| data path + compile + graph (`e91612c`) | 2.02–2.16x | 16.35 | 27 / 30 | 0.40 | 13.7 % | different |
| control: `lr x 1.02` from step 5 | 1.937x | **1864.96** | 28 / 30 | 2.51 | 47.1 % | different |
| reference, leave-one-out (positive control) | — | 2.19–6.10 | 0 / 10 per seed | 0.07 | 0.4 % | passes at 6.11 |

Recordings: reference 10 per seed, three seeds; each arm 1 per seed, three seeds. All in `bench/traj/`, with
each arm's judge report beside it. In absolute terms the deviations behind the z column are small: the
reference's own held-out runs move a statistic by a median of 7.7e-8 relative, the certified tree by 8.4e-8,
`round-1` by 4.3e-7 and the control arm by 2.8e-6.

This is the case's central result, and it is not the one the run's own work log claimed.

**The data-path work is numerically equivalent.** Its typical comparison is indistinguishable from the
reference's — median z 0.08 against 0.07 — and 0.07 % of its comparisons sit beyond 3 sigma, against the
reference's own 0.4 %. No step of any seed is outside the bound.

**`torch.compile` is not, and the gap is not marginal.** 13–15 % of its comparisons sit beyond 3 sigma —
seven to eight times the worst the reference ever produced in leave-one-out — and the median z is 5.5 times the
reference's. `round-2` is the same to two decimal places, and its CUDA graph never ran (section 6), so this is
the compiled blocks, not the graph. In absolute terms the shift is at most 2.9e-4 relative, but it is systematic
rather than a tail event, and a judge whose job is to separate an optimization from a different computation has
to call it.

**The control arm** is another two orders of magnitude out, which is the evidence that the gate separates
"different work" from "same work, different scheduling" rather than rejecting everything that is not bitwise
identical.

Four things about the measurement itself had to be fixed to reach that table, all recorded in
`task/judge_changes.md`: the judged statistic was initially a plain sum, which is ill-conditioned — cancellation
turned a 1e-6 difference into a 213 sigma reading; the bound was initially estimated from three repeats, which
does not pin sigma down; the threshold is calibrated against held-out reference runs instead of assumed at
3 sigma; and the loader was not carrying the parameter statistics the judge was reading, so the first
calibrated table compared only the loss scalars. That last defect is item 10 of section 4; it made the
compiled trees look marginal when they are not, and it is the reason the published threshold was 3.51 rather
than 6.11.

## 6. What the runs found

### The step was not compute-bound

Round 1's first ablation:

| variant | s/iter |
|---|---|
| full step | 0.368 |
| forward only | 0.368 |
| forward only, 750 MiB transfer skipped | 0.377 |
| **no model at all** (batch + trivial loss) | **0.384** |

The floor was host-side production of one dense `[4096, 47942]` float32 batch per iteration, 97.6 % of it
zeros. `cell_load`'s `_fetch_gene_expression_raw` built each cell as a one-row `sparse_csr_tensor(...).to_dense()`
— 2.81 ms/cell, 512 cells per sentence — and the collate ran two full-tensor scans whose result was discarded. A
NumPy scatter over the CSR arrays does the same work in 0.037 ms/cell, verified bitwise on 400 random cells.

### The gains

| step | s/iter | vs the reference tree |
|---|---|---|
| reference tree (`9bbfe78`) | 0.396 | 1.00 |
| + vectorised gradient norm | 0.368 | 1.08 |
| + CSR densify in the loader | 0.343 | 1.16 |
| + CSR batch shipped, densified on device | 0.212 | 1.87 |
| + RNG-safe compile of the backbone | 0.186 | 2.13 |
| + int32 columns, scatter straight into the buffer | 0.179 | 2.21 |

(The last column is against the reference tree's own 0.396, not against the frozen 0.3713; the harness verdicts
in section 5 are against live references, which is a third denominator. They are not interchangeable.)

### The CUDA graph in round 2 never ran

Round 2's work log attributes its gain to a CUDA graph capturing forward, both losses and backward. In every
timed run the capture failed and the tree fell back to eager, permanently:

```
[graphstep] capture failed, staying eager: RuntimeError: Attempting CUDA graph capture of
step() for an instance of Adam but param_groups' capturable is False.
```

The 5 % round 2 added over round 1 comes from the data-path commits, not the graph. The graph code is still in
the tree and still does nothing. This is exactly the class of error a judge has to catch by construction — the
agent's own account of its work is not evidence.

## 7. End-to-end validation

Both trees trained for 200 steps from the same seed (42) on the slice, through the same entrypoint, with the
optimized tree's speedups active. Per-step loss, reference vs candidate:

| | step 10 | step 50 | step 100 | mean, steps 150–199 |
|---|---|---|---|---|
| `train_loss`, reference | 3.9173 | 3.4455 | 3.1551 | 1.9672 |
| `train_loss`, candidate | 3.9173 | 3.4455 | 3.1549 | 1.9745 |
| `decoder_loss`, reference | 7.8730 | 3.3730 | 5.0768 | 3.1274 |
| `decoder_loss`, candidate | 7.8729 | 3.3730 | 5.0764 | 3.1213 |

Over the last 50 steps the two runs sit 0.37 % (`train_loss`) and 0.19 % (`decoder_loss`) apart — the scale of
the run-to-run variation this host produces, not a divergence. Training is not perturbed by the optimization.

## 8. Why it stops here

With the graph not running, the step is: eager `training_step` + a compiled backbone + the fast data path.
GPU kernels are ~131 ms of a ~178 ms step; the remainder is the eager clip and Adam (the harness's step counter
requires a real `optimizer.step()`), the 48 MB of CSR non-zeros over PCIe, and ~20 ms of host work.

Directions measured and rejected: Adam inside a graph (slower, breaks the step counter), the whole step in one
graph (~195 ms), compiling the whole forward (1.99x), `max-autotune`, and Adam `capturable=True` (10.97 vs
9.15 ms — and it is what the capture would need).

## 9. Reproducing

```bash
git clone https://github.com/YHDGD5/Faster-State && cd Faster-State
git clone https://github.com/arcinstitute/state.git targets/reference
git -C targets/reference checkout 9bbfe78a434a55205e4de834e1ea99f85f7a3add

# frozen inputs: 7.4 GB slice, fixture, gene panel (pinned by sha256 in task/formulation.json)
hf download YHDGD5/state-tahoe-case --repo-type dataset --local-dir assets
export ASSETS=$PWD/assets

python bench/bench.py check --final --tree ../state --out runs/verify
python bench/bench.py time        --tree ../state --out runs/t
```
