# Block-Masked-MAE-Improved-Self-Supervised-Pretraining-on-ViT-S-Tiny-IN

**Tiny-ImageNet • Vision Transformer • Masked Autoencoding • Texture Bias Reduction**

This repository contains the official implementation of **Block-Masked MAE**, a structure-oriented variant of Masked Autoencoders for Vision Transformers.  
Instead of random patch masking, we introduce **contiguous 2×2 / 4×4 block masking** to reduce texture shortcuts and force ViT to learn **semantic-level representations**.

---
## Highlights

- **Block-wise Masking (2×2 / 4×4)**  
  Encourages semantic reconstruction by removing fine-grained texture cues.

- **Semantic-vs-Texture Diagnostics**  
  Includes low-pass robustness, Stylized-ImageNet transfer, DTD texture dataset transfer, and occlusion-based attention rollout.

- **Long-run Pretraining (400–800 epochs)**  
  Full Tiny-ImageNet MAE training to test representation stability and semantic abstraction.

- **Reproducible Pipeline**  
  Config-driven, deterministic seeds, structured logs, automatic plotting, ablation modules, experiment registry.

- **Complete Ablation Suite**  
  Block size, mask ratio, patch size, decoder depth, positional embeddings.


---
## Project Structure

```bash
block-mae/
│── configs/                
│── data/                   
│── models/
│   ├── mae_vit_s.py        
│   └── mask_generator.py   
│── train/
│   ├── train_pretrain.py   
│   └── train_linear.py     
│── eval/
│   ├── lowpass.py          
│   ├── stylized_in.py      
│   ├── dtd_transfer.py     
│   └── attention_rollout.py
│── utils/                  
│── outputs/                
└── README.md


---

##  Method Overview

### 🔳 Block-wise Masking
Instead of independently sampling patch indices like MAE,  
we mask **contiguous spatial blocks**:

1×1 (standard MAE)
2×2 (ours)
4×4 (ours)

Block masks introduce **structured ambiguity**, making texture-based shortcuts harder and forcing the model to infer **object shape**.

---

## 📈 Experiments

### ✔ Pretraining Setup
| Setting | Value |
|--------|-------|
| Model | ViT-S (MAE) |
| Dataset | Tiny-ImageNet (SSL) |
| Epochs | 20, 50, 200, 400, 800 |
| Mask ratios | 65%, 75%, 85% |
| Block sizes | 1×1, 2×2, 4×4 |
| Optimizer | AdamW |
| Scheduler | Cosine Annealing |

---

## 🎯 Downstream Evaluation (Linear Probe)
We evaluate on:

- Tiny-ImageNet
- CIFAR-10
- CIFAR-100

Metrics:
- Top-1 accuracy
- Top-5 accuracy

(Replace below once results ready)

To be updated after long-run training.

yaml
Copy code

---

## 🧪 Ablations

### 1. Block Size
- 1×1 (MAE baseline)
- 2×2 (ours)
- 4×4 (ours)

### 2. Mask Ratio
- 65 / 75 / 85%

### 3. Decoder Depth
- 1 / 2 / 4 / 8 layers

### 4. Patch Size
- 16×16 vs 8×8

### 5. Positional Embedding
- Absolute
- Relative
- None (learned)

---

## 🔍 Semantic vs Texture Bias Analysis

### ① Low-pass Robustness  
Gaussian blur & FFT-based filtering to test shape sensitivity.

### ② Stylized ImageNet (SIN) Transfer  
Tests reliance on texture vs structure.

### ③ DTD Texture Dataset Transfer  
Texture-biased models perform excessively well;  
block-masked MAE should drop more → more semantic.

### ④ Occlusion-based Attention Rollout  
Stable attention = structure-based representation.

---

## 🖼️ Visualizations

### Reconstruction Samples
- Random mask vs Block mask
- 20 / 50 / 200 / 400 / 800 epoch comparison

### Attention Maps
- Attention rollout under occlusion  
- Center-of-mass tracking  
- Entropy heatmaps

---

## 🧩 How to Run

### Pretraining (MAE)
```bash
python train/train_pretrain.py --config configs/pretrain_block.yaml
Linear Probing
bash
Copy code
python train/train_linear.py --config configs/linear_probe.yaml
Low-pass Robustness
bash
Copy code
python eval/lowpass.py --checkpoint <path>
📦 Dependencies
PyTorch >= 2.1

timm

numpy / scipy

matplotlib

einops

pyyaml

tqdm

Install:

bash
Copy code
pip install -r requirements.txt
📄 Citation (Template)
bibtex
Copy code
@article{he2025blockmae,
  title={Block-Masked MAE: Structure-Oriented Self-Supervised Pretraining on Vision Transformers},
  author={He, Yuke},
  year={2025},
  note={Work in progress}
}
🗂️ Status
✅Block mask generator

✅20/50 epoch pilot experiments

✅Config-driven reproducible pipeline

 400–800 epoch long-run experiments

 Full semantic vs texture ablation

 Research draft (8–12 pages)

📨 Contact

He Yuke
GitHub: https://github.com/yokea1

Email: 217885@student.upm.edu.my
