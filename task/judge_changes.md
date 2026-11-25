# Judge change record (gates 3 and 4 added after both runs)

The judge was frozen before round 1 started and did not change during either run. It **did** change afterwards:
two gates were added once a review showed that the record it carried never covered the configuration gate 1
timed. This record gives the evaluator hash each verdict used, the hash that replaced it, and what changed, so
that a third party can judge whether the addition was a relaxation.

It was not. Both gates are additional checks on a candidate; neither loosens an existing one, and the tier the
case reports went **down** from N0 to N1 as a result.

## The evaluator hashes

| verdict | evaluator sha256 (first 12) | tree source_digest |
|---|---|---|
| round 1, 1.8848x | `65cec29c4023` | `825a070b8c0b8491…` — commit `4b0d91d` |
| round 2, 2.0153–2.0797x | `65cec29c4023` | `4e0a3577fcbaf938…` — commit `e91612c` |
| **certified, 1.6751x** | this version | `dac3e78d826b381f…` — commit `b438d76` |

`65cec29c4023` is the frozen judge both runs were measured by. The values of the runs belong to it, and remain
comparable with each other.

## What was wrong

`bench.py` recorded the numerics record with `--deterministic` always, and the candidate tree took a different
path when deterministic mode was on: the compiled backbone, the compiled decoder block and the whole-step CUDA
graph were all skipped. Gate 2 therefore recorded an **eager** step, while gate 1 timed a **compiled** one. The
record also replayed a frozen fixture through a standalone adapter (`state_adapter.py`), densifying it itself, so
the candidate's data pipeline — the three commits that took the step from 0.368 to 0.212 s/iteration — never
reached any gate.

Additionally the frozen launch path sets `torch.set_float32_matmul_precision("medium")` (TF32), while the record
ran at torch's default `highest`. The same 4096x2048x47942 GEMM measures 66.9 TFLOPS under `medium` and 18.4
under `highest`.

So the reported "794/794 tensors bit-identical, tier N0" was true of a configuration that was never timed.

## What changed

| | before | after |
|---|---|---|
| gate 2 (record, adapter) | carried the verdict | kept, as a record only: it covers the eager path |
| gate 3 (batches) | — | **new**, bitwise: each tree's own dataloader, same seed and sampler, 10 batches, every leaf tensor compared |
| gate 4 (trajectory) | — | **new**, statistical: `state tx train` at three seeds, 12 steps, judged against a bound frozen from the reference's own repeats |
| reported tier | N0 | **N1** |
| `check --final` verdict | gate 2's bitwise result | gates 3 and 4 jointly |

`bench/bench.json` gained `data` (batches, seed) and `path` (steps, seeds, repeats, bound). `bench/batch_check.py`
and `bench/traj_run.py` + `bench/traj_compare.py` are new. `evaluate()` in `bench/bench.py` runs the two new
gates only under `--final`, so a candidate's own `check` still gets a cheap answer.

## Why N1 and not N0

The timed path is not bitwise reproducible. Flash attention's backward accumulates with atomics: two runs of the
same tree at the same seed differ from step 2 on (measured parameter-sample deviation ~1e-4 relative, ~10x the
same-seed repeat spread by step 3). No bound on the timed configuration can be exact, so the case cannot claim
N0. It reports N1.

## The bound is fixed before the candidate is seen

`bench/traj_compare.py freeze` runs first, over at least three repeats of the reference per seed, and writes
`bench/traj/frozen.json`. `judge` then compares a candidate against that file. The multiplier is 3 sigma per
(step, component), the same 3 sigma the repository's own gate 3 uses. Components whose reference repeats agree
bitwise fall back to a scale of 1e-6 of their own magnitude, the round-off scale of a float32 accumulation.

The sketch the bound is computed over is 24 parameters spread evenly across the optimizer's parameter list
(backbone, decoder, head) plus every scalar the model logs per step. The first version sampled six tensors from
the start of the list, with no losses; a wrong step could have passed it.

## The bound's statistic (second change, same review)

The first version of gate 4 compared a plain **sum** of the first 64k elements of each sampled tensor. That
statistic is ill-conditioned: in float32 a sum of ~64k values with mixed signs can move by orders of magnitude
more than the tensor moved, purely through cancellation. Measured on the same recordings, the reference's own
same-seed spread on those sums was 1.7e-6 relative, while a candidate that only changed kernel order sat at
2.1e-4 — 120x the spread — and every legitimate optimization failed the 3-sigma bound.

The judged statistics are now the **L2 norm** and the **largest magnitude** of each sampled tensor, plus every
scalar the model logs. Both are immune to cancellation. The sum is still recorded, and still reported, as a
diagnostic only.

This is a change of *what is measured*, not of the threshold: the bound is still 3 sigma of the reference's own
repeats, still frozen before any candidate is recorded. The table below gives both statistics on the same arms,
so the effect of the change is visible rather than asserted.

| arm | sum-based z | norm-based z | verdict |
|---|---|---|---|
| certified tree (`b438d76`) | 213 | **1.86** | equivalent |
| round-1 tree (`4b0d91d`) | 213 | 4.14 | different (3 of 30 steps) |
| round-2 tree (`e91612c`) | 216 | 5.32 | different (4 of 30) |
| control, `lr x 1.02` from step 5 | 1865 | **1864.96** | different (19 of 30) |

The sum-based column is what the first version judged, and it is why the first version rejected every arm.

## The repeats (third change, same review)

Three repeats per seed leave sigma itself uncertain by ~50 %. The same arm measured 2.82 sigma in one
recording and 3.96 sigma in another -- 29 of 30 step-seeds inside the bound, one outside, the verdict flipping
on a single component. The bound now uses **ten repeats per seed** (thirty recordings, three seeds), which puts
that arm at 1.86 sigma with no step outside. The rule is unchanged: 3 sigma of the reference's own repeats,
frozen before any candidate is recorded.

## The threshold is calibrated, not assumed (fourth change, same review)

A per-comparison 3 sigma applied to the worst of ~1500 comparisons is not a 3 sigma event. Leave-one-out over
the 30 reference recordings -- freeze the bound on nine repeats of a seed, judge the tenth -- puts the
reference's own worst z anywhere from 0.94 to 3.50:

| seed | held-out reference worst z |
|---|---|
| 42 | 0.94 1.46 1.79 1.90 1.95 2.11 2.38 2.53 2.68 3.22 |
| 43 | 1.55 1.63 1.64 1.75 1.79 1.80 1.98 2.21 2.23 2.34 |
| 44 | 1.49 1.59 1.59 1.77 1.83 1.87 2.29 2.45 2.48 3.50 |

A full check draws one of these per seed and takes the largest, so at 3.0 the **reference passes only 81 % of
its own checks**. The frozen threshold is the smallest value at which that three-seed draw passes 95 % of the
time: **3.51**. `bench/traj/frozen.json` carries the whole leave-one-out table under `calibration`, and the
reference is reported as the gate's positive control next to the `lr x 1.02` arm.

## The batch gate was not working

`bench/build_fixture.py` imported `cell_load` but not the tree's own `state` package, and the data-path patch is
installed at `import state`. The gate therefore drew batches through the **unpatched** loader on both sides and
reported them identical -- a check that could not fail. It now imports the tree's package first, and compares
semantically: a candidate that ships the gene counts as a CSR bundle has its dense tensor rebuilt before the
comparison. With the patch active the gate still passes -- 10 batches, 163,900 leaf tensors, zero differences --
which is the first real evidence that the data path is value-preserving.

## Regression

The control arm is the optimized tree with the learning rate multiplied by 1.02 — faster *and* wrong, so it
reaches gate 4 in a full check rather than being stopped at gate 1. It is recorded in
`bench/control_arm.json` and the verdict is in the case submission.
