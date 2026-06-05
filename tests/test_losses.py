"""Known-value tests for loss functions: DiceLoss, dice_bce_loss, weighted_mse_loss, make_loss."""

import pytest
import torch
from omegaconf import OmegaConf

from inr_unet.losses import DiceLoss, dice_bce_loss, make_loss, weighted_mse_loss


def test_dice_loss_near_zero_on_perfect_prediction():
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    logits = (target * 2 - 1) * 20.0  # sigmoid -> ~target
    assert DiceLoss()(logits, target).item() < 1e-2


def test_dice_loss_near_one_on_disjoint_prediction():
    target = torch.zeros(2, 1, 16, 16)
    logits = torch.full((2, 1, 16, 16), 20.0)  # predict all foreground
    assert DiceLoss()(logits, target).item() > 0.9


def test_dice_loss_finite_and_optimal_on_all_zero():
    target = torch.zeros(2, 1, 8, 8)
    logits = torch.full((2, 1, 8, 8), -20.0)  # predict all background
    loss = DiceLoss()(logits, target).item()
    assert loss == loss and loss < 1e-2  # not NaN, near 0


def test_dice_loss_works_on_query_shape():
    target = (torch.rand(2, 100, 1) > 0.5).float()
    logits = (target * 2 - 1) * 20.0
    assert DiceLoss()(logits, target).item() < 1e-2


def test_dice_bce_reduces_to_bce_at_zero_weight():
    import torch.nn.functional as F

    logits = torch.randn(2, 1, 8, 8)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    combined = dice_bce_loss(logits, target, dice_weight=0.0)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    assert torch.allclose(combined, bce)


def test_weighted_mse_zero_on_perfect_prediction():
    target = torch.rand(2, 1, 8, 8)  # soft target in [0, 1]
    logits = torch.log(target.clamp(1e-4, 1 - 1e-4) / (1 - target.clamp(1e-4, 1 - 1e-4)))
    assert weighted_mse_loss(logits, target).item() < 1e-6


def test_weighted_mse_penalizes_foreground_more_than_background():
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 0, 0] = 1.0  # single foreground peak
    # both wrong pixels are set to logit=0 (sigmoid=0.5), giving equal prob-space errors;
    # only the weight differs
    base = torch.log(target.clamp(1e-4, 1 - 1e-4) / (1 - target.clamp(1e-4, 1 - 1e-4)))
    fg_wrong = base.clone()
    fg_wrong[0, 0, 0, 0] = 0.0  # sigmoid 0.5 vs target 1.0
    bg_wrong = base.clone()
    bg_wrong[0, 0, 1, 1] = 0.0  # sigmoid 0.5 vs target 0.0
    assert weighted_mse_loss(fg_wrong, target).item() > weighted_mse_loss(bg_wrong, target).item()


def test_weighted_mse_gradient_drives_peak_up():
    target = torch.zeros(1, 1, 3, 3)
    target[0, 0, 1, 1] = 1.0
    logits = torch.zeros(1, 1, 3, 3, requires_grad=True)
    weighted_mse_loss(logits, target).backward()
    assert logits.grad[0, 0, 1, 1].item() < 0  # increasing the peak logit lowers the loss


def test_weighted_mse_works_on_query_shape():
    target = torch.rand(2, 100, 1)
    logits = torch.log(target.clamp(1e-4, 1 - 1e-4) / (1 - target.clamp(1e-4, 1 - 1e-4)))
    assert weighted_mse_loss(target=target, logits=logits).item() < 1e-6


def test_make_loss_selects_weighted_mse():
    cfg = OmegaConf.create({"train": {"loss": "weighted_mse", "fg_weight": 10.0}})
    loss_fn = make_loss(cfg)
    logits = torch.randn(1, 1, 4, 4)
    target = torch.rand(1, 1, 4, 4)
    expected = weighted_mse_loss(logits, target, fg_weight=10.0)
    assert torch.allclose(loss_fn(logits, target), expected)


def test_make_loss_selects_dice_bce():
    cfg = OmegaConf.create({"train": {"loss": "dice_bce", "fg_weight": 10.0}})
    loss_fn = make_loss(cfg)
    logits = torch.randn(1, 1, 4, 4)
    target = (torch.rand(1, 1, 4, 4) > 0.5).float()
    assert torch.allclose(loss_fn(logits, target), dice_bce_loss(logits, target))


def test_make_loss_unknown_raises():
    cfg = OmegaConf.create({"train": {"loss": "nope", "fg_weight": 10.0}})
    with pytest.raises(ValueError):
        make_loss(cfg)
