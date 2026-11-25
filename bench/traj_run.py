"""Record a short training trajectory by running the tree's real entrypoint.

Unlike `numerics record`, which drives the model through a standalone adapter, this goes through
`state tx train` exactly as the timing run does: the same dataloader, the same compiled and captured step,
the same matmul precision. That is what makes the trajectory a check on the configuration gate 1 times
instead of on an eager shadow of it.

Per optimizer step it stores, from a fixed sample of the optimizer's parameters, a sha256 over their first
elements (a bitwise signal) and their float64 sums (a statistical one). `traj_compare.py` decides which of the
two the run earns.

usage: python bench/traj_run.py --tree <tree> --out <dir> --steps 12 --seed 42 [--deterministic] [-- <overrides>]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

SAMPLE_TENSORS = 6
SAMPLE_ELEMS = 1 << 16


class StopTraj(BaseException):
    """Ends the run from inside the step hook. A BaseException so the training code's `except Exception`
    cannot swallow it."""


def sketch(params):
    h = hashlib.sha256()
    sums = []
    for t in params:
        flat = t.detach().reshape(-1)
        n = min(flat.numel(), SAMPLE_ELEMS)
        v = flat[:n].float().cpu().numpy()
        h.update(v.tobytes())
        sums.append(float(v.astype('float64').sum()))
    return h.hexdigest()[:16], sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--warmup', type=int, default=2, help='steps recorded but marked warm-up (graph capture, compile)')
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--sample-tensors', type=int, default=SAMPLE_TENSORS)
    ap.add_argument('--losses', action='store_true', help='also record the scalars the model logs')
    ap.add_argument('overrides', nargs='*')
    a = ap.parse_args()

    tree = Path(a.tree).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(tree / 'src'))
    os.chdir(tree)

    if a.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    st = {'k': 0, 'steps': [], 't0': time.perf_counter(), 'error': None, 'logs': {}}

    if a.losses:
        from lightning.pytorch import LightningModule
        _orig_log = LightningModule.log

        def _log(self, name, value, *args, **kwargs):
            try:
                v = value.detach().float().mean().item() if torch.is_tensor(value) else float(value)
                if v == v:                       # skip nan
                    st['logs'].setdefault(st['k'], {})[name] = v
            except (TypeError, ValueError, RuntimeError):
                pass
            return _orig_log(self, name, value, *args, **kwargs)

        LightningModule.log = _log

    def hook(opt, *_args, **_kwargs):
        params = opt.param_groups[0]['params'][:a.sample_tensors]
        digest, sums = sketch(params)
        st['steps'].append({'step': st['k'], 'warmup': st['k'] < a.warmup,
                            'digest': digest, 'sums': sums,
                            'losses': st['logs'].get(st['k'], {}),
                            't': round(time.perf_counter() - st['t0'], 4)})
        st['k'] += 1
        if st['k'] >= a.steps:
            raise StopTraj()

    register_optimizer_step_post_hook(hook)

    sys.argv = ['state', *a.overrides, f'training.train_seed={a.seed}']
    status = 'ok'
    try:
        from state.__main__ import main as state_main
        state_main()
    except StopTraj:
        pass
    except BaseException as exc:                                   # noqa: BLE001
        import traceback
        status, st['error'] = 'error', f'{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}'

    rec = {'schema': 'bench-trajectory@1', 'tree': str(tree), 'seed': a.seed, 'steps': st['steps'],
           'deterministic': a.deterministic, 'status': status, 'error': st['error'],
           'warmup': a.warmup, 'torch': torch.__version__}
    (out / f'traj.s{a.seed}.r0.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({'status': status, 'seed': a.seed, 'steps': len(st['steps']),
                      'last_digest': st['steps'][-1]['digest'] if st['steps'] else None}))
    return 0 if status == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
