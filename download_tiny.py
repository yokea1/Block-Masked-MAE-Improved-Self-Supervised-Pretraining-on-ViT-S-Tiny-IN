"""
download_tiny.py
-----------------
自动下载 Tiny-ImageNet 数据集。
"""

import os
import urllib.request
import zipfile

url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
save = "tiny-imagenet-200.zip"

if not os.path.exists(save):
    print("[INFO] Downloading Tiny-ImageNet...")
    urllib.request.urlretrieve(url, save)

print("[INFO] Extracting...")
with zipfile.ZipFile(save, "r") as zip_ref:
    zip_ref.extractall(".")
print("[INFO] Done!")
