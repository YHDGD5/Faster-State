# State (Arc Institute) — Tahoe-100M training step: a frozen-judge optimization case

A ResearchBoost-Agent case built around one training step of the [State](https://github.com/arcinstitute/state)
transition model on a slice of [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M). The judge is
frozen; a candidate is judged on being faster *and* numerically identical *and* on the same code path.

This repository is the case package. The large frozen assets live on Hugging Face — see **Assets** below.

## Result

| round | tree | candidate s/iter | live reference in that check | speedup | tier |
|---|---|---|---|---|---|
| 1 — cold, unmodified start | `4b0d91d` | 0.1883 | 0.355 | **1.885x** | N1 |
| 2 — warm, from round 1's tree | `e91612c` | 0.1797–0.1789 | 0.360–0.369 | **2.02–2.08x** | N1 |

Three denominators appear in this repository and they are not interchangeable:

- **live reference in that check** — two runs of `targets/reference` timed in the same session. This is what gate
  1 divides by, and it is the column above. The reference moves 11 % between runs on this shared host.
- **0.3713 s/iteration** — the frozen `task/timing_ref.json`, recorded once at formulation. Against it the same
  candidates read 1.97x and 2.07–2.09x.
- **0.3956 s/iteration** — the unmodified tree, measured at the time of the run. Against it the same candidates
  read 2.10x and 2.21–2.23x.

The headline number is the harness's own, which is the first column's ratio.

Both rounds are reported separately on purpose: round 2 begins from a tree that already carries round 1's
changes and from a workspace that contains round 1's findings. The cold number is the one that measures an
unassisted agent; the warm number measures how much headroom is left once the obvious work is done.

## The workload

| | |
|---|---|
| Upstream | `github.com/arcinstitute/state` @ `9bbfe78` |
| Config | `tahoe_best` (`cell_set_len=512`) |
| Step | batch 8 cell sentences x 512 cells = 4096 cell rows |
| Parameters | 209.1 M |
| Hardware | 1 x A800 80 GB PCIe, `world_size=1` |
| Baseline | 0.3713 s / iteration (frozen); 209.1 M params; Adam |

The slice is 13 plates / 50 cell lines / 380 drugs / 1243 samples — 399,990 cells x 47,942 genes, built with a
per-shard quota so the batch is not dominated by a single plate.

## The judge

```
python bench/bench.py check --tree <candidate> --out <dir>
```

| gate | question |
|---|---|
| 0 structure | did the tree change something that makes the measurement meaningless? |
| 1 benefit | is the candidate faster than **two live reference runs** in the same session? |
| 2 record | are the recorded tensors of the **eager** path bit-identical to the reference? (a record, not the verdict) |
| 3 batches | does the candidate's own dataloader produce the reference's batches, element for element? |
| 4 trajectory | does the candidate's real training trajectory stay inside a bound frozen from the reference's own repeats? |

Gates 3 and 4 run the configuration gate 1 times, and they carry the verdict. They were added after both runs,
once a review showed gate 2 recorded an eager step while gate 1 timed a compiled one — see
[`task/judge_changes.md`](task/judge_changes.md). The tier the case reports is **N1**: the timed path is not
bitwise reproducible, because flash attention's backward accumulates with atomics.

Timing: 30 warm-up iterations, then 3 windows of 40, judged on window means; bands are derived from the
measured 17–20 % run-to-run noise of a shared host. `bench/`, `numerics/`, `task/` and `targets/` are
read-only in the run's sandbox, and the harness refuses a verdict if any of them changed since launch.

The judge was tested against three known-wrong candidates before it was trusted:

| candidate | expected | observed |
|---|---|---|
| unmodified reference | no benefit | `UNAVAILABLE` / `no_benefit`, 0.9384x |
| **control arm on the optimized tree, `lr x 1.02`** | gate 4 | _pending_ |
| control arm on the reference, `lr x 1.02` | gate 1 (1.03x is not a speedup) | `UNAVAILABLE` / `no_benefit` |
| tampered frozen file | gate 0 | `ERROR` at structure |
| pristine reference | pass | `PASS` |

## Assets

The judge's inputs are too large for git. Download them from the Hugging Face dataset
[`YHDGD5/state-tahoe-case`](https://huggingface.co/datasets/YHDGD5/state-tahoe-case) and place them here:

```
assets/data/dataset/tahoe_slice.h5ad      7.4 GB   the 13-plate slice
assets/data/gene_metadata.parquet         1.3 MB   gene panel
bench/fixture.pt                          1.8 GB   frozen numerics fixture (6 batches)
```

```bash
bash fetch_assets.sh          # downloads from HF and verifies SHA256SUMS
```

`task/formulation.json` pins these by content, so a mismatch fails the judge before anything runs.

## Running it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e <path-to-state> -r requirements.txt   # torch 2.15, lightning, geomloss, cell_load

bash fetch_assets.sh

python bench/bench.py check  --tree state --out runs/verify     # the judge
python bench/bench.py time   --tree state --out runs/t          # timing only
python bench/bench.py profile --tree state --out runs/profile   # torch profiler + trace
```

## What the optimization found

The step was **not** compute-bound. A step with no model at all — just the batch and a trivial loss — took
0.384 s/iteration, the same as the full model. The floor was host-side production of one dense
`[4096, 47942]` float32 batch per iteration (749 MiB, 97.6 % zeros): `cell_load` built each cell as a
one-row `sparse_csr_tensor(...).to_dense()` at 2.81 ms/cell, and the collate ran two full-tensor scans whose
result was discarded.

| step | s/iter | gain |
|---|---|---|
| baseline | 0.396 | — |
| vectorised gradient norm | 0.368 | 1.08x |
| CSR densify in `fetch_gene_expression` | 0.343 | 1.16x |
| CSR batch, densify on device | 0.212 | 1.87x |
| RNG-safe `torch.compile` | 0.186 | 2.04x |

Full detail, including the measured and rejected directions, is in [`docs/case-report.md`](docs/case-report.md)
and `WORK_LOG.round1.md`.

## Layout

```
bench/        the judge: bench.py, recipe.py, timing.py, state_adapter.py, bench.json,
              records/ref (2 steps x 397 tensors), thresholds.json (794 gates),
              batch_check.py and traj_run.py/traj_compare.py (the two gates on the timed path),
              traj/ (the frozen reference trajectories)
numerics/     the numerics tier machinery (record, trajectory, judge, stats)
task/         formulation.json (input fingerprints), timing_ref.json (0.3713 s/iter)
prep/         rebuilds the slice from the raw Tahoe-100M shards
docs/         case-report.md
GOAL.md       the task as given to the optimization agent
```

## Attribution

State — Arc Institute (`github.com/arcinstitute/state`, MIT/Apache per its LICENSE).
Tahoe-100M — Tahoe Bio, distributed under its own dataset license. This repository contains only a derived
slice (cell selection and gene filtering) plus the judge machinery.
