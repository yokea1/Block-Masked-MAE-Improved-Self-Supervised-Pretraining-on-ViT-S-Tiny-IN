"""
block_mask.py
---------------
MAE 改进版：2x2 block mask （改进自己的想法）
"""
import torch
from utils import device

class BlockMask:
    def __init__(self, num_patches, block=2, ratio=0.75):
        self.num_patches = num_patches
        self.block = block
        self.ratio = ratio

    def __call__(self, B):
        mask = torch.zeros(B, self.num_patches).bool().to(device)
        blocks = self.num_patches // (self.block * self.block)
        num_mask = int(blocks * self.ratio)

        for i in range(B):
            ids = torch.randperm(blocks)[:num_mask]
            for bid in ids:
                base = bid * (self.block * self.block)
                mask[i, base : base + self.block*self.block] = True

        return mask
