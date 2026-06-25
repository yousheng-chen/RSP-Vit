# RSP-ViT: Recurrent Shared-weight Patch Vision Transformer
## Micro-Doppler Radar Target Recognition with Shared-weight ViT

**RSP-ViT** (Recurrent Shared-weight Patch Vision Transformer) is a lightweight Vision Transformer architecture for micro-Doppler radar spectrogram classification. It introduces two key design choices: **cross-layer weight sharing** (inspired by ALBERT) to reduce parameter count, and a **ResNet-style convolutional stem** to replace the standard patch embedding. Together these form a parameter-efficient ViT tailored for radar recognition tasks.

---

## Key Features

- **Weight-shared Transformer encoder** — All N layers share the same attention and FFN parameters, drastically reducing model size while retaining depth.
- **ResNet-style patch stem** — A lightweight convolutional front-end (conv_stem_channels=[32, 64]) extracts local texture before mapping to patch tokens, outperforming direct linear patch embedding.
- **Comprehensive ablations** — Systematically compares 4 ViT variants across 5 noise levels: Vanilla ViT, Shared ViT, Stem-only ViT, and RSP-ViT (Shared + ResNet Stem).
- **7 CNN baselines** — All trained under the same protocol for fair comparison: ResNet18/34/50, EfficientNet-B0, GoogLeNet, MobileNetV2, MobileNetV3-Small, MnasNet1.0, ShuffleNetV2.
- **Noise robustness evaluation** — Trains and evaluates on datasets with additive Gaussian noise at 5 levels (-10, -5, 0, +5, +10 dB).

---

## Project Structure

`
.
├── src/
│   ├── training.py                         # Core ViT training loop
│   ├── config.py                           # Default hyper-parameters
│   ├── resnet_train.py                     # ResNet family training
│   ├── official_cnn_train.py               # Unified CNN training entry
│   ├── efficientnetb0_train.py             # EfficientNet-B0 launcher
│   ├── googlenet_train.py                  # GoogLeNet launcher
│   ├── mobilenetv2_train.py                # MobileNetV2 launcher
│   ├── mobilenetv3_small_train.py          # MobileNetV3-Small launcher
│   ├── mnasnet1_0_train.py                 # MnasNet1.0 launcher
│   ├── mnasnet1_0_manual_train.py          # MnasNet1.0 manual impl.
│   ├── shufflenetv2_train.py               # ShuffleNetV2 launcher
│   ├── run_paper_ablation_benchmark.py     # Main paper ablation
│   ├── run_vit_noise_benchmark.py          # Shared vs unshared ViT
│   ├── run_share_depth_repeat.py           # Depth embedding ablation
│   ├── run_share_localffn_repeat.py        # FFN variant ablation
│   ├── run_share_vs_resnet_stem_benchmark.py
│   ├── run_share_vs_shufflenet_benchmark.py
│   ├── run_share_vs_mobilenetv2_benchmark.py
│   ├── run_share_vs_mobilenetv2_fair_benchmark.py
│   ├── run_share_vs_mobilenetv3_small_benchmark.py
│   ├── run_share_vs_efficientnet_benchmark.py
│   ├── run_share_vs_googlenet_benchmark.py
│   ├── run_share_vs_mnasnet_benchmark.py
│   ├── run_share_vs_mnasnet1_0_manual_benchmark.py
│   ├── noise_sweep_shared.py               # Online noise robustness
│   ├── noise_sweep_collect.py              # Collect run results
│   ├── compare_runs.py                     # Run comparison
│   ├── models/
│   │   ├── vit.py                          # VisionTransformer model
│   │   ├── transformerlayer.py             # Transformer encoder layer
│   │   ├── mha.py                          # Multi-Head Attention
│   │   ├── ffn.py                          # FeedForward variants
│   │   └── *_official.py                   # CNN constructors
│   └── dataset/
│       ├── dataloader.py                   # ImageFolder dataloader
│       ├── processing_data.py              # Spectrogram preprocessing
│       └── split_dataset.py                # Dataset splitting
├── overleaf_springer_ready/                # LaTeX paper (Springer LNCS)
├── requirements.txt
└── README.md
`

---

## Requirements

`
torch >= 1.12
torchvision >= 0.13
matplotlib
numpy
scikit-learn
tqdm
`

Install: pip install -r requirements.txt

---

## Usage

### Train RSP-ViT
`ash
python src/training.py --data-dir D:/micro_doppler_ai-main/data/noisy/-10db --num-epochs 10 --batch-size 32 --learning-rate 0.0001 --share-transformer-weights --patch-embed-variant resnet_stem
`

### Run the paper ablation (all 4 ViT variants across noise levels)
`ash
python src/run_paper_ablation_benchmark.py --data-root D:/micro_doppler_ai-main/data/noisy --num-epochs 10 --batch-size 32
`

### Compare against a CNN baseline
`ash
python src/run_share_vs_mobilenetv2_benchmark.py --data-root D:/micro_doppler_ai-main/data/noisy --num-epochs 10
`

### Train individual CNN baselines
`ash
python src/resnet_train.py
python src/efficientnetb0_train.py
python src/official_cnn_train.py --model-name shufflenet_v2_x1_0
`

### Noise robustness sweep
`ash
python src/noise_sweep_shared.py --data-root D:/micro_doppler_ai-main/data/noisy --noise-levels 0 5 10 15 20
`

---

## Default Model Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| n_heads | 8 |
| n_layers | 6 |
| patch_size | 16 |
| d_ff | 1024 |
| img_size | 192x192 |
| batch_size | 32 |
| learning_rate | 1e-4 |
| weight_decay | 1e-5 |
| conv_stem_channels | [32, 64] |

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

If you use this code in your research, please cite:

`ibtex
@misc{chen2026rspvit,
  title={RSP-ViT: Recurrent Shared-weight Patch Vision Transformer for Micro-Doppler Radar Target Recognition},
  author={Yousheng Chen and ...},
  year={2026},
}
`
