"""
visualize_block.py
--------------------
可视化 MAE 重建（block-mask）
"""

from utils import device
import torch
from PIL import Image
import matplotlib.pyplot as plt
from mae_model_block import MAE_Block
from torchvision import transforms

img = Image.open("tiny-imagenet-200/val/images/val_0.JPEG")

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

img_tensor = transform(img).unsqueeze(0).to(device)

model = MAE_Block().to(device)
model.load_state_dict(torch.load("mae_tinysmall_block.pth", map_location=device))
model.eval()

recon, mask = model(img_tensor)

recon = recon.reshape(1, -1, 16, 16, 3).permute(0, 4, 2, 3, 1)
recon_img = recon.reshape(1, 3, 224, 224)[0].detach().cpu().permute(1,2,0)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title("Original"); plt.imshow(img_tensor[0].cpu().permute(1,2,0))
plt.subplot(1,3,2); plt.title("Masked");   plt.imshow(img_tensor[0].cpu().permute(1,2,0))
plt.subplot(1,3,3); plt.title("Reconstructed (Block)");
plt.imshow(recon_img)
plt.show()
