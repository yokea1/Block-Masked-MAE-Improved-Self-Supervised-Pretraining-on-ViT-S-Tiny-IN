"""
train_pretrain_block.py
-------------------------
训练 MAE（block-mask + ViT-Small + Tiny-ImageNet）
（备注：不用了 因为太慢了 算力匮乏 现在在用 train_pretrain_block.fast.py版本 留在这儿原因是为了对比没加速前和加速后的成果）

"""

from utils import device
import torch
import torch.nn as nn
from mae_model_block import MAE_Block
from tiny_loader import get_tiny_loader
import matplotlib.pyplot as plt

train_loader, _ = get_tiny_loader(batch=32)

model = MAE_Block().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

loss_list = []

EPOCHS = 50     # M1 推荐训练量20 但是为了好的结果用了50 （算力不足 m1芯片跑了 1hour但是我改进了用了m1特化加速版）
for epoch in range(EPOCHS):
    for imgs, _ in train_loader:
        imgs = imgs.to(device)

        recon, mask = model(imgs)

        recon = recon.reshape(imgs.size(0), -1, 16, 16, 3)
        recon = recon.permute(0, 4, 2, 3, 1)
        recon = recon.reshape(imgs.size(0), 3, 224, 224)

        loss = criterion(recon, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch}] Loss = {loss.item():.4f}")
    loss_list.append(loss.item())

torch.save(model.state_dict(), "mae_tinysmall_block.pth")
plt.plot(loss_list)
plt.title("MAE Block-Mask (Tiny-IN + ViT-S)")
plt.savefig("loss_block.png")
