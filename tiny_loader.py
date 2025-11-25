"""
tiny_loader.py
-----------------
Tiny-ImageNet 加载器
64×64 → 224×224
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_tiny_loader(batch=32):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    train = datasets.ImageFolder("tiny-imagenet-200/train", transform=transform)
    val = datasets.ImageFolder("tiny-imagenet-200/val", transform=transform)

    return (
        DataLoader(train, batch_size=batch, shuffle=True),
        DataLoader(val, batch_size=batch)
    )
