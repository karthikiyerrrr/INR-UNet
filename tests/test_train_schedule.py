"""Optimizer build + warmup/cosine LR lambda behave as specified."""

import torch
from omegaconf import OmegaConf

from inr_unet.train import build_optimizer, make_lr_lambda


def _cfg(scheduler="cosine", warmup_frac=0.2, lr=1e-3, weight_decay=1e-4):
    return OmegaConf.create(
        {"train": {"lr": lr, "weight_decay": weight_decay,
                   "scheduler": scheduler, "warmup_frac": warmup_frac}}
    )


def test_build_optimizer_is_adamw_with_wd():
    model = torch.nn.Linear(2, 2)
    opt = build_optimizer(model, _cfg())
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == 1e-3
    assert opt.param_groups[0]["weight_decay"] == 1e-4


def test_cosine_warms_up_then_decays():
    fn = make_lr_lambda(_cfg(scheduler="cosine", warmup_frac=0.2), total_steps=10)
    assert fn(0) < fn(1)          # warmup ramps (warmup_steps = 2)
    assert fn(1) == 1.0           # peak at end of warmup
    assert fn(9) < fn(2)          # decays afterwards
    assert fn(9) >= 0.0


def test_none_scheduler_is_constant():
    fn = make_lr_lambda(_cfg(scheduler="none"), total_steps=10)
    assert fn(0) == 1.0 and fn(5) == 1.0 and fn(9) == 1.0
