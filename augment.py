"""Data augmentations for Snake conv-encoder inputs.

CURL applies the same random crop across all frames in a stack
("time-consistent spatial jittering"). For a 20x20 board we pad with zeros
and crop back, which is the natural grid analogue of the paper's
random crop from 100x100 to 84x84.
"""

import torch
import torch.nn.functional as F


def random_pad_and_crop(x: torch.Tensor, pad: int = 2) -> torch.Tensor:
    """Vectorized random pad-and-crop.

    Args:
        x: (B, C, H, W) float tensor. The C axis spans all stacked frames,
           so the same crop is applied to every frame of every sample (the
           crop is shared per-sample, varied across the batch).
        pad: zero-padding added on each side before cropping.

    Returns:
        (B, C, H, W) tensor with the same shape as the input.
    """
    if pad <= 0:
        return x
    B, C, H, W = x.shape
    padded = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)
    # Sample one (top, left) offset per batch element in [0, 2*pad].
    tops = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    lefts = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    out = torch.empty_like(x)
    for i in range(B):
        t, l = int(tops[i].item()), int(lefts[i].item())
        out[i] = padded[i, :, t:t + H, l:l + W]
    return out


def identity(x: torch.Tensor) -> torch.Tensor:
    return x
