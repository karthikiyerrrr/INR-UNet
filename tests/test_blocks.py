"""Contract tests for the UNet conv blocks: shapes, channels, gradient flow."""

import torch

from inr_unet.models.components.blocks import DoubleConv, Down, ResBlock, Up


def test_double_conv_preserves_spatial_changes_channels():
    block = DoubleConv(3, 8)
    out = block(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 8, 16, 16)


def test_down_halves_spatial():
    block = Down(8, 16)
    out = block(torch.randn(2, 8, 16, 16))
    assert out.shape == (2, 16, 8, 8)


def test_up_doubles_and_concats_skip():
    block = Up(16, skip_channels=8, out_channels=8)
    deep = torch.randn(2, 16, 8, 8)
    skip = torch.randn(2, 8, 16, 16)
    out = block(deep, skip)
    assert out.shape == (2, 8, 16, 16)


def test_up_aligns_odd_sized_skip():
    block = Up(16, skip_channels=8, out_channels=8)
    deep = torch.randn(2, 16, 7, 9)      # up -> 14x18
    skip = torch.randn(2, 8, 15, 19)     # mismatched by 1 in each dim
    out = block(deep, skip)
    assert out.shape == (2, 8, 15, 19)


def test_res_block_is_shape_preserving():
    block = ResBlock(12)
    out = block(torch.randn(2, 12, 10, 10))
    assert out.shape == (2, 12, 10, 10)


def test_blocks_have_gradients():
    block = DoubleConv(3, 8)
    out = block(torch.randn(2, 3, 16, 16))
    out.sum().backward()
    grads = [p.grad is not None for p in block.parameters()]
    assert all(grads) and len(grads) > 0
