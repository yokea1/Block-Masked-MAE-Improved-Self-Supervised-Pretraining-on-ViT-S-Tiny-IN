"""
mae_model_block.py
-------------------
MAE 模型（增强版）：
- 使用 ViT-Small
- 使用 block-wise masking（创新点）
"""

from utils import device
import torch
import torch.nn as nn
import timm
from block_mask import BlockMask


class MAE_Block(nn.Module):
    def __init__(self, img_size=224, mask_ratio=0.75):
        super().__init__()

        self.mask_ratio = mask_ratio

        # 使用更强的 backbone：ViT-Small
        self.encoder = timm.create_model("vit_small_patch16_224", pretrained=False)
        embed_dim = self.encoder.embed_dim

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.patch_size = 16
        self.num_patches = (img_size // 16) ** 2

        # block mask
        self.block_mask = BlockMask(self.num_patches, block=2, ratio=mask_ratio)

        # 投影回像素
        self.proj = nn.Linear(embed_dim, self.patch_size * self.patch_size * 3)

    def forward(self, imgs):
        B = imgs.size(0)

        patches = self.encoder.patch_embed(imgs)  # (B, N, D)
        B, N, D = patches.shape

        # mask
        mask = self.block_mask(B)
        visible = patches[~mask].reshape(B, -1, D)

        # encode
        encoded = self.encoder.blocks(self.encoder.pos_drop(visible))

        # decode
        decoded = self.decoder(encoded)

        # full patch embedding
        full = torch.zeros(
            B, N, decoded.size(-1),
            dtype=decoded.dtype,
            device=device
        )
        full[~mask] = decoded.reshape(-1, decoded.size(-1))

        # project to pixels
        patch_pixels = self.proj(full)  # (B, N, 768)

        # reshape patches back to image
        patch_size = self.patch_size  # =16
        h = w = int((self.num_patches) ** 0.5)  # =14×14 grid

        patch_pixels = patch_pixels.reshape(B, h, w, 3, patch_size, patch_size)
        patch_pixels = patch_pixels.permute(0, 3, 1, 4, 2, 5)
        # (B, 3, H, P, W, P) → (B, 3, 224, 224)
        recon = patch_pixels.reshape(B, 3, h * patch_size, w * patch_size)

        return recon, mask
