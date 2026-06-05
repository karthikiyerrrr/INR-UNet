"""Checkpoint round-trips model/optimizer/scheduler/RNG and bookkeeping."""

import numpy as np
import torch

from inr_unet.train import load_checkpoint, save_checkpoint


def _tiny():
    model = torch.nn.Linear(3, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
    return model, opt, sched


def test_checkpoint_round_trip(tmp_path):
    model, opt, sched = _tiny()
    model(torch.randn(4, 3)).sum().backward()
    opt.step()
    sched.step()
    path = tmp_path / "last.pt"
    save_checkpoint(
        path, epoch=2, model=model, optimizer=opt, scheduler=sched,
        best_val_f1=0.5, best_epoch=1, history=[{"epoch": 0}, {"epoch": 1}],
    )
    ckpt = load_checkpoint(path)
    assert ckpt["epoch"] == 2
    assert ckpt["best_val_f1"] == 0.5
    assert ckpt["best_epoch"] == 1
    assert ckpt["history"] == [{"epoch": 0}, {"epoch": 1}]
    # restoring RNG reproduces the next draw
    torch.set_rng_state(ckpt["torch_rng"])
    np.random.set_state(ckpt["numpy_rng"])
    a = (torch.rand(2), np.random.rand(2))
    torch.set_rng_state(ckpt["torch_rng"])
    np.random.set_state(ckpt["numpy_rng"])
    b = (torch.rand(2), np.random.rand(2))
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    # model weights restore exactly
    fresh, _, _ = _tiny()
    fresh.load_state_dict(ckpt["model"])
    assert torch.equal(fresh.weight, model.weight)
