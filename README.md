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
