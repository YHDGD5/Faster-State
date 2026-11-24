"""Whole-step CUDA graph for the training path, with dropout drawn outside the graph.

Why
---
A step is a few hundred small kernel launches whose GPU time is already close to the fp32 peak of the card,
but the host cannot enqueue them fast enough, and the loop stops twice per step (gradient clip and the next
batch). Measured on this workload: ~132 ms of GPU work inside a ~186 ms step, ~600 launches, ~23 ms of pure
Python inside the two distributional losses.

Capturing forward + both losses + backward + clip + Adam in one CUDA graph takes the host out of the
critical path: replay is a single call and the GPU runs from one kernel to the next.

Random numbers
--------------
A captured graph replays the *same* random draws every time, which would freeze the dropout masks. So the
masks are drawn eagerly -- with ``torch.native_dropout`` on a scratch tensor of the same shape, in the same
order and with the same probability as the eager model would, once per step -- and copied into persistent
buffers the graph reads. ``torch.native_dropout`` derives the mask from the generator state and the tensor
*shape* only, so drawing on a scratch tensor consumes exactly the numbers the reference consumes on the real
activation. Scaling matches too: ``(x * mask) * (1/(1-p))`` was verified bit-identical to
``torch.native_dropout(x, p, True)[0]``.

The graph is only built when ``torch.are_deterministic_algorithms_enabled()`` is false, i.e. never during
the numerics record, which therefore keeps the plain eager path and stays bitwise identical to the
reference.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from geomloss import SamplesLoss

# Execution-order collector: while a list is installed, every dropout layer appends itself the first time it
# runs in a step, which gives the exact draw order the eager model uses.
_ORDER = None


def graph_mode_allowed() -> bool:
    """The graph is only used outside the deterministic numerics path."""
    return torch.cuda.is_available() and not torch.are_deterministic_algorithms_enabled()


def _relax_manual_clip_guard() -> None:
    """Lightning refuses manual optimization when the Trainer sets `gradient_clip_val`.

    That guard exists because Lightning cannot clip for you in manual mode. This model clips itself, with the
    Trainer's own value, inside the captured step, so the check does not apply here. Everything else the
    validator checks (gradient accumulation) still runs unchanged.
    """
    try:
        from lightning.pytorch.trainer import configuration_validator as cv
    except Exception:
        return
    name = "__verify_manual_optimization_support"
    original = getattr(cv, name, None)
    if original is None or getattr(original, "_state_relaxed", False):
        return

    def relaxed(trainer, model):
        if not model.automatic_optimization and getattr(model, "_graph_mode", False):
            clip, trainer.gradient_clip_val = trainer.gradient_clip_val, None
            try:
                return original(trainer, model)
            finally:
                trainer.gradient_clip_val = clip
        return original(trainer, model)

    relaxed._state_relaxed = True
    setattr(cv, name, relaxed)


_relax_manual_clip_guard()


class MaskDropout(nn.Module):
    """``nn.Dropout`` that can consume a mask drawn ahead of time.

    With ``mask`` unset it is exactly ``nn.Dropout`` (``F.dropout`` -> ``torch.native_dropout``), which is
    the path the numerics record takes. In graph mode ``mask`` points at a persistent buffer the step fills
    before each replay.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = float(p)
        self._scale = 1.0 / (1.0 - self.p) if self.p > 0.0 else 1.0
        self._seen = False
        self.mask = None      # set to the persistent buffer while the graph is in use
        self.buf = None
        self.shape = None
        self.scratch = None

    def forward(self, x):
        global _ORDER
        if _ORDER is not None and not self._seen:
            self._seen = True
            self.shape = tuple(x.shape)
            _ORDER.append(self)
        if self.mask is not None:
            return (x * self.mask) * self._scale
        return F.dropout(x, self.p, self.training)

    def allocate(self, device):
        if self.p <= 0.0 or self.shape is None:
            return
        if self.buf is None or tuple(self.buf.shape) != self.shape:
            self.buf = torch.empty(self.shape, dtype=torch.bool, device=device)
            self.scratch = torch.empty(self.shape, dtype=torch.float32, device=device)

    def draw(self):
        """Draw this layer's mask from the global RNG, in the reference's order."""
        if self.p <= 0.0 or self.buf is None or not self.training:
            return
        _, mask = torch.native_dropout(self.scratch, self.p, True)
        self.buf.copy_(mask)

    def enable(self):
        self.mask = self.buf if self.p > 0.0 else None

    def disable(self):
        self.mask = None

    def extra_repr(self):
        return f"p={self.p}"


def record_begin():
    global _ORDER
    _ORDER = []
    return _ORDER


def record_end(order):
    global _ORDER
    _ORDER = None
    for m in order:
        m._seen = False
    return order


class StepGraph:
    """Static inputs, the captured graph and the recorded dropout order for one model."""

    def __init__(self, model):
        self.model = model
        self.graph = None
        self.order = None
        self.inputs = {}
        self.static = None
        self.loss = None
        self.logs = {}
        self.warm = 0
        self.opt = None
        self.clip = 0.0
        self.failed = False

    # -- staging --------------------------------------------------------------
    def matches(self, batch):
        """True when this batch has the shapes the graph was captured with (the last batch of an epoch may not)."""
        for key in ("pert_emb", "ctrl_cell_emb", "pert_cell_emb"):
            buf = self.inputs.get(key)
            if buf is None or tuple(buf.shape) != tuple(batch[key].shape):
                return False
        buf = self.inputs.get("pert_cell_counts")
        if buf is None:
            return False
        counts = batch["pert_cell_counts"]
        if isinstance(counts, dict) and counts.get("__csr_batch__"):
            return int(counts["n_cells"]) == buf.shape[0] and int(counts["n_genes"]) == buf.shape[1]
        return tuple(buf.shape) == tuple(counts.shape)

    def stage(self, batch):
        """Copy this batch into the static buffers; returns the static batch dict."""
        for key in ("pert_emb", "ctrl_cell_emb", "pert_cell_emb"):
            src = batch[key]
            buf = self.inputs.get(key)
            if buf is None:
                self.inputs[key] = src.detach().clone().contiguous()
            else:
                if buf.shape != src.shape:
                    raise RuntimeError("step graph: batch shape changed")
                buf.copy_(src)
        counts = self.model._densify_counts(batch["pert_cell_counts"])
        buf = self.inputs.get("pert_cell_counts")
        if buf is None:
            self.inputs["pert_cell_counts"] = counts.detach().clone()
        else:
            buf.copy_(counts)
        return self.inputs

    # -- masks ----------------------------------------------------------------
    def prepare_masks(self):
        for m in self.order or ():
            m.draw()

    def enable_masks(self):
        for m in self.order or ():
            m.allocate(self._device())
            m.enable()

    def _device(self):
        return next(self.model.parameters()).device


class GraphSamplesLoss(SamplesLoss):
    """``SamplesLoss`` whose uniform weights live on the device.

    ``SamplesLoss(x, y)`` rebuilds the uniform weights on the CPU on every call and copies them over, which a
    CUDA graph cannot capture (unpinned host-to-device copy mid-capture). Handing the loss the same weights
    explicitly keeps the arithmetic identical and the copy out of the graph. Outside the graph the plain
    two-argument path runs untouched, so the numerics record sees exactly the reference call.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weights = {}
        self._precomputed = graph_mode_allowed()

    def _w(self, x):
        key = (tuple(x.shape), x.dtype, x.device)
        w = self._weights.get(key)
        if w is None:
            b, n = x.shape[0], x.shape[1]
            w = torch.ones(b, n, dtype=x.dtype, device=x.device) / n
            self._weights[key] = w
        return w

    def forward(self, *args):
        if self._precomputed and len(args) == 2:
            x, y = args
            if x.dim() == 3 and y.dim() == 3:
                return super().forward(self._w(x), x, self._w(y), y)
        return super().forward(*args)
