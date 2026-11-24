import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from torch.optim import Optimizer

from ..models import PerturbationModel
from .batch_speed_monitor import BatchSpeedMonitorCallback
from .model_flops_utilization import ModelFLOPSUtilizationCallback
from .cumulative_flops import CumulativeFLOPSCallback

__all__ = ["PerturbationModel", "BatchSpeedMonitorCallback", "ModelFLOPSUtilizationCallback", "CumulativeFLOPSCallback"]


class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """

    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer
    ) -> None:
        pl_module.log("train/gradient_norm", gradient_norm(pl_module))


def gradient_norm(model):
    """L2 norm of all gradients, computed on the device in one shot.

    The reference accumulates ``p.grad.norm(2).item()`` per parameter, which forces one host sync per
    parameter tensor (79 of them on this model) and stalls the GPU against the host every iteration.
    Collecting the per-tensor norms with a single fused ``_foreach_norm`` and reducing them in float64
    leaves exactly one sync per step, and the value is only ever logged (gradient clipping is done by
    Lightning), so the training math is unchanged.
    """
    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    norms = torch._foreach_norm(grads, 2)
    # Match the reference's float64 accumulation of squared fp32 per-tensor norms.
    total_norm = torch.stack([n.to(torch.float64) for n in norms]).pow(2).sum()
    return float(total_norm.sqrt().item())
