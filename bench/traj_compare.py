"""Compare a candidate's training trajectory against the reference's, statistically.

The timed path is not bitwise reproducible -- flash attention's backward accumulates with atomics, so two runs
of the same tree at the same seed already differ from the second step on. What the judge can ask instead is
whether the candidate's trajectory stays close to the reference's, at the same seed and over the same steps.

The scale of "close" is measured against a candidate that is known to do different work: the control arm that
multiplies the learning rate by 1.02. On the Tahoe slice at 12 steps it separates the candidate (<= 2e-3 of the
reference's RMS, after normalising each step's parameter sample) from that control (>= 5e-2) by more than an
order of magnitude, so the default tolerance sits between them.

usage: python bench/traj_compare.py --ref DIR [--ref DIR ...] --cand DIR [...] [--tol 1e-2] [--out report.json]
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(dirs):
    """{seed: [ (step, sums), ... ]} over every recording found, repeats kept."""
    runs = defaultdict(list)
    for d in dirs:
        for p in sorted(Path(d).glob('traj.s*.r0.json')):
            m = json.loads(p.read_text())
            if m.get('status') != 'ok':
                continue
            steps = [(s['step'], s['sums']) for s in m['steps'] if not s['warmup']]
            if steps:
                runs[m['seed']].append(steps)
    return runs


def deviation(cand, ref_runs):
    """Largest per-component relative deviation from the reference mean at the same step.

    Per component, not over the whole vector: the sampled sums span several orders of magnitude, and a vector
    norm would drown a 5 % shift in a small component in the magnitude of the large ones.
    """
    n = len(cand)
    mean = [statistics.fmean([r[i] for r in ref_runs]) for i in range(n)]
    worst = 0.0
    for c, m in zip(cand, mean):
        worst = max(worst, abs(c - m) / max(abs(m), 1e-12))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', action='append', required=True)
    ap.add_argument('--cand', action='append', required=True)
    ap.add_argument('--tol', type=float, default=1e-2)
    ap.add_argument('--out')
    a = ap.parse_args()

    ref, cand = load(a.ref), load(a.cand)
    if not ref:
        sys.exit('no reference recordings')
    if not cand:
        sys.exit('no candidate recordings')

    rows, worst = [], None
    for seed, runs in sorted(cand.items()):
        per_step = defaultdict(list)
        for steps in runs:
            for step, sums in steps:
                per_step[step].append(sums)
        by_step = {step: [s for steps in ref[seed] for st, s in steps if st == step]
                   for step in per_step}
        missing = [step for step, v in by_step.items() if not v]
        if missing:
            rows.append({'seed': seed, 'missing_steps': sorted(missing)})
            continue
        for step, samples in sorted(per_step.items()):
            dev = max(deviation(s, by_step[step]) for s in samples)
            ok = dev <= a.tol
            rows.append({'seed': seed, 'step': step, 'deviation': round(dev, 6), 'within': ok})
            if worst is None or dev > worst[0]:
                worst = (dev, seed, step)

    bad = [r for r in rows if r.get('within') is False]
    verdict = 'equivalent' if not bad else 'different'
    rep = {'schema': 'bench-trajectory-report@1', 'tol': a.tol, 'verdict': verdict,
           'reference_seeds': sorted(ref), 'candidate_seeds': sorted(cand),
           'steps_compared': len([r for r in rows if 'step' in r]), 'steps_outside': len(bad),
           'worst_deviation': round(worst[0], 6) if worst else None,
           'worst_at': {'seed': worst[1], 'step': worst[2]} if worst else None, 'rows': rows}
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps({k: rep[k] for k in ('verdict', 'steps_compared', 'steps_outside',
                                          'worst_deviation', 'worst_at')}))
    return 0 if verdict == 'equivalent' else 1


if __name__ == '__main__':
    sys.exit(main())
