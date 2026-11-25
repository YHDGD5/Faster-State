"""Freeze a trajectory bound from the reference's own repeats, then judge a candidate against it.

The timed path is not bitwise reproducible -- flash attention's backward accumulates with atomics -- so the
question is whether a candidate stays inside the reference's own run-to-run spread. That spread is measured
from repeats of the reference at the same seed, and the bound is 3 sigma of it per (step, component), fixed
*before* any candidate is recorded.

  freeze:  python bench/traj_compare.py freeze --ref DIR [--ref DIR ...] --out bench/traj/frozen.json
  judge:   python bench/traj_compare.py judge  --frozen bench/traj/frozen.json --cand DIR [...] [--out r.json]

A component's sigma can be zero when the reference repeats agree bitwise; there the scale falls back to 1e-6 of
the component's own magnitude, which is the round-off scale of a float32 accumulation, so a genuine difference
still shows up and a representation change does not.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

K = 3.0            # sigma multiplier

# Logged scalars that are diagnostics rather than outputs: step timing, flop counters, and the gradient norm,
# which the first round deliberately computes differently for logging. Judging them would reject an
# optimization for changing a log line, not a computation. They are still recorded and still reported.
DIAGNOSTIC = {'batch_time_avg', 'batch_time_cv_percent', 'batch_time_max', 'batch_time_max_min_ratio',
              'batch_time_min', 'batches_per_second', 'cumulative_flops', 'train/gradient_norm'}
REL_FLOOR = 1e-6   # scale floor when the reference repeats agree bitwise


def load(dirs):
    """{seed: [ {step: {'sums': [...], 'losses': {...}}}, ... ]} -- one entry per recording."""
    runs = defaultdict(list)
    for d in dirs:
        for p in sorted(Path(d).glob('traj.s*.r0.json')):
            m = json.loads(p.read_text())
            if m.get('status') != 'ok':
                continue
            steps = {s['step']: {'sums': s['sums'], 'losses': s.get('losses') or {}}
                     for s in m['steps'] if not s['warmup']}
            if steps:
                runs[m['seed']].append(steps)
    return runs


def components(entry, include_sums=False):
    """Flatten one step into {name: value}.

    The judged statistics are the L2 norm and the largest magnitude of each sampled tensor, plus every scalar
    the model logs. A plain sum of a 64k-element slice is ill-conditioned -- cancellation moves it by orders of
    magnitude more than the tensor itself moved -- so it is carried only as a diagnostic (`include_sums`).
    """
    out = {}
    for i, v in enumerate(entry.get('norms') or []):
        out[f'norm{i}'] = v
    for i, v in enumerate(entry.get('absmax') or []):
        out[f'absmax{i}'] = v
    if include_sums:
        for i, v in enumerate(entry.get('sums') or []):
            out[f'sum{i}'] = v
    for k, v in sorted((entry.get('losses') or {}).items()):
        if k not in DIAGNOSTIC:
            out[f'loss/{k}'] = v
    return out


def freeze(ap):
    runs = load(ap.ref)
    if not runs:
        sys.exit('no reference recordings')
    thin = {s: len(v) for s, v in runs.items() if len(v) < 3}
    for s, n in sorted(thin.items()):
        print(f'  seed {s}: {n} repeat(s), fewer than 3 -- skipped, no spread to measure', file=sys.stderr)
    frozen = {'schema': 'bench-trajectory-frozen@1', 'k': K, 'rel_floor': REL_FLOOR,
              'skipped_seeds': {str(s): n for s, n in thin.items()}, 'seeds': {}}
    for seed, reps in sorted(runs.items()):
        if len(reps) < 3:
            continue
        steps = sorted(set.intersection(*[set(r) for r in reps]))
        per_step = {}
        for st in steps:
            cols = defaultdict(list)
            for r in reps:
                for name, v in components(r[st]).items():
                    cols[name].append(v)
            per_step[str(st)] = {n: {'mean': statistics.fmean(v), 'sigma': statistics.stdev(v)}
                                 for n, v in sorted(cols.items())}
        frozen['seeds'][str(seed)] = {'repeats': len(reps), 'steps': per_step}
        worst = max((e['sigma'] / max(abs(e['mean']), 1e-30) for s in per_step.values() for e in s.values()),
                    default=0.0)
        print(f'  seed {seed}: {len(reps)} repeats, {len(steps)} steps, worst sigma/|mean| {worst:.3e}',
              file=sys.stderr)
    Path(ap.out).write_text(json.dumps(frozen, indent=1) + '\n')
    print(json.dumps({'frozen': ap.out, 'seeds': sorted(int(s) for s in frozen['seeds'])}))
    return 0


def judge(ap):
    frozen = json.loads(Path(ap.frozen).read_text())
    k, floor = frozen['k'], frozen['rel_floor']
    cand = load(ap.cand)
    if not cand:
        sys.exit('no candidate recordings')

    rows, worst = [], None
    for seed, reps in sorted(cand.items()):
        ref = frozen['seeds'].get(str(seed))
        if not ref:
            rows.append({'seed': seed, 'error': 'no frozen reference for this seed'})
            continue
        mag = {}
        for ref_step_m in ref['steps'].values():
            for name, e in ref_step_m.items():
                mag[name] = max(mag.get(name, 0.0), abs(e['mean']))
        for i, run in enumerate(reps):
            for st, entry in sorted(run.items()):
                ref_step = ref['steps'].get(str(st))
                if not ref_step:
                    rows.append({'seed': seed, 'repeat': i, 'step': st, 'error': 'step not in the frozen bound'})
                    continue
                z, detail = 0.0, None
                for name, v in components(entry).items():
                    e = ref_step.get(name)
                    if e is None:
                        continue
                    # Per component, never scaled by another component's magnitude: the components span twelve
                    # orders of magnitude (a parameter sum vs cumulative_flops), and a shared scale would drown
                    # a real difference in the small ones. The floor uses this component's own typical size
                    # over the recorded steps, so a component that sits near zero every step still has a
                    # sane bound, and one that is identically zero in the reference requires exact equality.
                    scale = max(e['sigma'], floor * mag.get(name, 0.0))
                    if scale <= 0.0:
                        scale = float('inf') if v == e['mean'] else 0.0
                    zi = abs(v - e['mean']) / scale if scale else float('inf')
                    if zi > z:
                        z, detail = zi, {'component': name, 'value': v, 'ref_mean': e['mean'],
                                         'ref_sigma': e['sigma']}
                rows.append({'seed': seed, 'repeat': i, 'step': st, 'z': round(z, 2),
                             'within': z <= k, 'worst_component': detail})
                if worst is None or z > worst[0]:
                    worst = (z, seed, st, detail)

    bad = [r for r in rows if r.get('within') is False]
    verdict = 'equivalent' if not bad else 'different'
    rep = {'schema': 'bench-trajectory-report@2', 'k': k,
           'candidate': [str(Path(d).resolve()) for d in ap.cand],
           'verdict': verdict, 'steps_compared': len([r for r in rows if 'step' in r]),
           'steps_outside': len(bad),
           'worst_z': round(worst[0], 2) if worst else None,
           'worst_at': {'seed': worst[1], 'step': worst[2], 'component': (worst[3] or {}).get('component')}
           if worst else None,
           'rows': rows}
    if ap.out:
        Path(ap.out).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps({x: rep[x] for x in ('verdict', 'steps_compared', 'steps_outside', 'worst_z', 'worst_at')}))
    return 0 if verdict == 'equivalent' else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--ref', action='append', required=True)
    f.add_argument('--out', required=True)
    j = sub.add_parser('judge')
    j.add_argument('--frozen', required=True)
    j.add_argument('--cand', action='append', required=True)
    j.add_argument('--out')
    a = ap.parse_args()
    return freeze(a) if a.cmd == 'freeze' else judge(a)


if __name__ == '__main__':
    sys.exit(main())
