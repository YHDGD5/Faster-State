"""Do the candidate's dataloader and the reference's produce the same batches?

The recipe allows the data pipeline to change "as long as the batches are identical", and nothing used to
check that: the numerics adapter replayed a frozen fixture and densified it itself, so `fast_fetch`, the CSR
batch and the on-device densify never reached a gate. This draws N batches from each tree's own dataloader
with the same seed and sampler, and compares them element by element.

usage: python bench/batch_check.py --tree <candidate> --toml <dataset.toml> --batches 10 [--out report.json]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import recipe  # noqa: E402


def draw(tree, toml, n, seed, out):
    cmd = [sys.executable, str(HERE / 'build_fixture.py'), '--tree', str(tree), '--toml', str(toml),
           '--out', str(out), '--batches', str(n), '--seed', str(seed),
           '--batch-size', str(recipe.BATCH_SIZE), '--cell-set-len', str(recipe.CELL_SET_LEN)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tree))
    if r.returncode or not Path(out).exists():
        return None, (r.stderr or r.stdout)[-1500:]
    return torch.load(out, map_location='cpu', weights_only=False), None


def leaves(obj, path=''):
    """Walk dicts, lists and tensors down to (path, tensor) pairs."""
    if torch.is_tensor(obj):
        yield path, obj.to_dense() if obj.is_sparse else obj
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            yield from leaves(obj[k], f'{path}/{k}')
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from leaves(v, f'{path}[{i}]')
    else:
        yield path, obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', required=True)
    ap.add_argument('--toml', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--batches', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out')
    a = ap.parse_args()
    a.toml = str(Path(a.toml).resolve())

    with tempfile.TemporaryDirectory(prefix='statebatch-') as tmp:
        ref, err = draw(Path(a.reference).resolve(), a.toml, a.batches, a.seed, Path(tmp) / 'ref.pt')
        if ref is None:
            sys.exit(f'the reference dataloader failed: {err}')
        cand, err = draw(Path(a.tree).resolve(), a.toml, a.batches, a.seed, Path(tmp) / 'cand.pt')
        if cand is None:
            sys.exit(f'the candidate dataloader failed: {err}')

    rows, worst = [], None
    for i, (rb, cb) in enumerate(zip(ref['ref'], cand['ref'])):
        lr, lc = dict(leaves(rb)), dict(leaves(cb))
        for k in sorted(set(lr) | set(lc)):
            if k not in lr or k not in lc:
                rows.append({'batch': i, 'key': k, 'equal': False, 'reason': 'missing'})
                continue
            x, y = lr[k], lc[k]
            if not (torch.is_tensor(x) and torch.is_tensor(y)):
                rows.append({'batch': i, 'key': k, 'equal': bool(x == y), 'max_abs_diff': 0.0})
                continue
            if x.shape != y.shape:
                rows.append({'batch': i, 'key': k, 'equal': False,
                             'reason': f'shape {tuple(x.shape)} vs {tuple(y.shape)}'})
                continue
            eq = torch.equal(x, y)
            d = 0.0
            if not eq:
                if x.is_floating_point():
                    d = float((x.double() - y.double()).abs().max())
                else:
                    d = float((x != y).sum())
                if worst is None or d > worst[0]:
                    worst = (d, i, k)
            rows.append({'batch': i, 'key': k, 'equal': eq, 'max_abs_diff': d})

    bad = [r for r in rows if not r['equal']]
    rep = {'schema': 'bench-batches@1', 'seed': a.seed, 'batches': len(ref['ref']),
           'tensors_compared': len(rows), 'tensors_differing': len(bad),
           'equal': not bad, 'worst': {'max_abs_diff': worst[0], 'batch': worst[1], 'key': worst[2]} if worst else None,
           'rows': rows}
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps({k: rep[k] for k in ('equal', 'batches', 'tensors_compared', 'tensors_differing', 'worst')}))
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
