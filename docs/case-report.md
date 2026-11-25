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
backward accumulates with atomics, so two runs of the *same* tree at the same seed differ from the second step
on — so the question is whether the candidate stays within a measured tolerance of the reference. The tolerance
is calibrated against a candidate known to do different work:

| | worst relative deviation (30 step-seeds) |
|---|---|
| the candidate | 6.7e-3 |
| control arm, learning rate x 1.02 | 2.3e-1 |

The tolerance is 1e-2, which sits in that gap. This control arm is a calibration of the tolerance, not an
end-to-end gate rejection: in a full check it is stopped earlier — at the benefit gate, since a 2 % learning
rate change is not a measurable speedup, and on the record path by its `invariants`. Anyone repeating the check
can reproduce the 2.3e-1 with `bench/traj_compare.py` over that arm's recordings.

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

## 5. Results

Two optimization runs, then a third tree built to be certifiable. Only the last one carries the case's number.

| tree | commit | source_digest | candidate | speedup |
|---|---|---|---|---|
| cold run | `4b0d91d` | `825a070b8c0b8491…` | 0.1883 s/iter | 1.885x |
| warm run | `e91612c` | `4e0a3577fcbaf938…` | 0.1789 s/iter | 2.02–2.08x |
| **certified** | **`b438d76`** | **`dac3e78d826b381f…`** | **0.2215 s/iter** | **1.675x** |

The first two are what the two goal runs reached. The third is the same data-path work with `torch.compile`
turned off — two lines — and it is the one the trajectory gate certifies. The speed is lower than what the runs
achieved; section 5's table below is why.

Three denominators appear and are not interchangeable: the **live reference** timed in the same session (what
gate 1 divides by; 0.355–0.386 s/iteration, 11 % spread on this shared host), the **frozen** 0.3713
(`task/timing_ref.json`), and the **unmodified tree** measured at the time (0.3956). 0.3956 / 0.1883 is 2.10x;
the 1.885x divides by that check's live reference.

### What the trajectory gate says about each tree

The gate records `state tx train` at three seeds for 12 steps and compares it against a bound frozen from
**ten repeats of the reference per seed** — 3σ per (step, component), written before any candidate was recorded.
The judged statistics are the L2 norm and the largest magnitude of 24 sampled parameter tensors, plus the
model's loss scalars. Step timing, flop counters and the logged gradient norm are recorded but not judged: they
are diagnostics, and the first round deliberately computes the gradient norm differently for logging.

| tree | speedup | worst deviation | steps outside 3σ | verdict |
|---|---|---|---|---|
| **data path only, no compile (`b438d76`)** | **1.675x** | **1.86σ** | **0 / 30** | **equivalent** |
| data path + `torch.compile` (`4b0d91d`) | 1.885–1.934x | 4.14σ | 3 / 30 | different |
| data path + compile + graph (`e91612c`) | 2.02–2.16x | 5.32σ | 4 / 30 | different |
| control: `lr x 1.02` from step 5 | 1.937x | **1864.96σ** | 19 / 30 | different |

This is the case's central result, and it is not the one the run's own work log claimed.

**The data-path work is numerically equivalent.** It passes with no step outside the bound.

**`torch.compile` is not.** It lands 4.1–5.3σ out, beyond the reference's own run-to-run spread. Reordering the
backbone's reductions moves the parameters more than repeating the reference at the same seed does. So the case
certifies the slower tree.

The control arm fails by three orders of magnitude more, which is what makes the gate a gate.

Two things about the measurement itself had to be fixed to reach that table, and both are recorded in
`task/judge_changes.md`: the judged statistic was initially a plain sum, which is ill-conditioned — cancellation
turned a 1e-6 difference into a 213σ reading; and the bound was initially estimated from three repeats, which
does not pin σ down — the certified tree measured 2.82σ in one recording and 3.96σ in another, flipping on a
single step-seed out of thirty.

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
