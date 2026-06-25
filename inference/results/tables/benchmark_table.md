# Deployment Complexity Analysis

| Model | Params (M) | FLOPs (G) | Latency (ms) | GPU Mem (MB) | Ckpt (MB) | Train Time (s) | Test Acc (%) | Best Val (%) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Vanilla ViT | 6.28 | 0.07 | 13.91 | -- | 24.01 | 460 | 98.83 | 99.61 |
| Shared ViT | 2.34 | 0.07 | 15.37 | -- | 8.93 | 443 | 96.06 | 95.89 |
| Stem-only ViT | 6.37 | 0.19 | 15.77 | -- | 24.34 | 562 | 100.00 | 99.89 |
| RSP-ViT (Ours) | 2.42 | 0.19 | 13.79 | -- | 9.26 | 737 | 99.89 | 99.89 |