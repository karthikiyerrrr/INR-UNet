"""Known-value tests for DiceLoss and dice_bce_loss."""

import torch

from inr_unet.losses import DiceLoss, dice_bce_loss


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
