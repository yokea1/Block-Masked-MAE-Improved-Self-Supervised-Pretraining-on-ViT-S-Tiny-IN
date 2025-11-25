"""
utils.py
---------
统一管理设备选择（MPS / CUDA / CPU）。
M1/M2 芯片自动使用 Apple GPU (mps)。 我的设备是macbook m1芯片
"""
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"[INFO] Using device: {device}")
