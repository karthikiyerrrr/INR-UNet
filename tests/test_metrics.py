"""Known-value tests for iou, dice_score, pixel_accuracy."""

import torch

from inr_unet.metrics import dice_score, iou, pixel_accuracy


def test_iou_perfect_and_disjoint():
    target = (torch.rand(1, 1, 16, 16) > 0.5).float()
    assert iou(target, target) == 1.0
    assert iou(1.0 - target, target) == 0.0


def test_iou_empty_is_one():
    z = torch.zeros(1, 1, 8, 8)
    assert iou(z, z) == 1.0


def test_dice_score_perfect_and_disjoint():
    target = (torch.rand(1, 1, 16, 16) > 0.5).float()
    assert dice_score(target, target) == 1.0
    assert dice_score(1.0 - target, target) == 0.0


def test_pixel_accuracy_perfect_and_half():
    target = torch.zeros(1, 1, 4, 4)
    assert pixel_accuracy(target, target) == 1.0
    half = target.clone()
    half[..., :2, :] = 1.0  # half the pixels wrong
    assert pixel_accuracy(half, target) == 0.5
