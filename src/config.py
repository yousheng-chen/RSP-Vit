import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#比较share与share—depth
#C:/Users/13023/.conda/envs/vit/python.exe src/run_share_depth_repeat.py --benchmark-name share_depth_minus10db_repeat2 --data-dir D:/micro_doppler_ai-main/data/noisy/-10db --repeats 2 --num-epochs 10 --batch-size 32 --learning-rate 0.0001 --weight-decay 1e-5 --img-size 192 192 --train-split 0.5 --val-split 0.25 --seed 42


#C:/Users/13023/.conda/envs/vit/python.exe src/resnet_train.py
#C:/Users/13023/.conda/envs/vit/python.exe src/training.py
#C:/Users/13023/.conda/envs/vit/python.exe src/resnet_train_presplit.py


#对比share与unshare，run_vit_noise_benchmark.py   2.335  win
#C:/Users/13023/.conda/envs/vit/python.exe src/run_vit_noise_benchmark.py --data-root D:/micro_doppler_ai-main/data/noisy --checkpoint-root D:/micro_doppler_ai-main/checkpoint --num-epochs 5 --batch-size 32 --learning-rate 0.0001


#对比unshare与ShuffleNetV2，run_vit_noise_benchmark.py   1.263   win
#C:/Users/13023/.conda/envs/vit/python.exe src/run_share_vs_shufflenet_benchmark.py --data-root D:/micro_doppler_ai-main/data/noisy --checkpoint-root D:/micro_doppler_ai-main/checkpoint --benchmark-name share_vs_shufflenet --num-epochs 10 --batch-size 32 --learning-rate 0.0001 --weight-decay 1e-5 --img-size 192 192 --train-split 0.5 --val-split 0.25

#对比unshare与EfficientNet-B0    4.019  lose
#C:/Users/13023/.conda/envs/vit/python.exe src/run_share_vs_efficientnet_benchmark.py --data-root D:/micro_doppler_ai-main/data/noisy --checkpoint-root D:/micro_doppler_ai-main/checkpoint --benchmark-name share_vs_efficientnet --num-epochs 10 --batch-size 32 --learning-rate 0.0001 --weight-decay 1e-5 --img-size 192 192 --train-split 0.5 --val-split 0.25

#bug 对比unshare与MnasNet1.0，   3，113，841  bug no
#C:/Users/13023/.conda/envs/vit/python.exe src/run_share_vs_mnasnet1_0_manual_benchmark.py --benchmark-name share_vs_mnasnet1_0_manual --resume --share-source-dirs "share vs unshare" "share_vs_shufflenet" "share_vs_efficientnet" --num-epochs 10 --batch-size 32 --learning-rate 0.0001 --weight-decay 1e-5 --img-size 192 192 --train-split 0.5 --val-split 0.25

#对比unshare与MobileNetV2        2，234，401  win
#最佳C:/Users/13023/.conda/envs/vit/python.exe src/run_share_vs_mobilenetv2_benchmark.py --benchmark-name share_vs_mobilenetv2 --resume
#重新训练可改训练集：C:/Users/13023/.conda/envs/vit/python.exe src/run_share_vs_mobilenetv2_fair_benchmark.py --benchmark-name share_vs_mobilenetv2_fair --resume --num-epochs 10 --batch-size 32 --learning-rate 0.0001 --weight-decay 1e-5 --img-size 192 192 --train-split 0.5 --val-split 0.25


#对比unshare与mobilenet_v3_small   1，527    win
#C:/Users/13023/.conda/envs/vit/python.exe src/run_share_vs_mobilenetv3_small_benchmark.py --benchmark-name share_vs_mobilenetv3_small --resume --num-epochs 10 --batch-size 32 --learning-rate 0.0001 --weight-decay 1e-5 --img-size 192 192 --train-split 0.5 --val-split 0.25

#对比unshare与googlenet    5,69   lose
#C:/Users/13023/.conda/envs/vit/python.exe src/run_share_vs_googlenet_benchmark.py --benchmark-name share_vs_googlenet --resume --num-epochs 10 --batch-size 32 --learning-rate 0.0001 --weight-decay 1e-5 --img-size 192 192 --train-split 0.5 --val-split 0.25

#C:/Users/13023/.conda/envs/vit/python.exe src/training.py --share-transformer-weights --ffn-variant first_layer_local_enhanced



#RegNetY_400MF，googlenet，

config = {
    # ViT data/training config
    'data_dir': r"D:\micro_doppler_ai-main\data\noisy\-10db",
    'img_size': [192, 192],
    'batch_size': 32,

    # ViT model config
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 6,
    'patch_size': 16,
    'd_ff': 1024,
    'share_transformer_weights': True,

    'use_depth_embeddings': False,#加一个层身份

    'patch_embed_variant': 'resnet_stem',#前端引用resnet改进
    'conv_stem_channels': [32, 64],

    'ffn_variant': 'standard',#增强ffn1

# patch_embed_variant 可以在 standard 和 resnet_stem 之间切换。
# resnet_stem 会先用轻量卷积前端提取局部纹理，再映射成 Transformer token。

# ffn_variant可以在 standard、local_enhanced 和 first_layer_local_enhanced 之间切换(无用)。
# first_layer_local_enhanced 表示只有第 1 层使用局部增强 FFN，后面层保持标准 FFN。


#C:/Users/13023/.conda/envs/vit/python.exe src/training.py
#C:/Users/13023/.conda/envs/vit/python.exe src/run_share_localffn_repeat.py --benchmark-name share_localffn_minus10db --data-dir D:/micro_doppler_ai-main/data/noisy/-10db --repeats 2 --num-epochs 10 --batch-size 32 --learning-rate 0.0001 --weight-decay 1e-5 --img-size 192 192 --train-split 0.5 --val-split 0.25 --seed 42

    # ViT training config
    'num_epochs': 10,
    'learning_rate': 0.0001,
    'weight_decay': 1e-5,
    'seed': 41,
    'split_seed': 42,

    # ViT outputs
    'checkpoint_dir': "checkpoint",
}


ResNet18_config = {
    # Dataset config
    'data_dir': r"D:\micro_doppler_ai-main\data\noisy\-10db",
    'img_size': [192, 192],
    'batch_size': 32,

    # Dataset split config
    # Example: 0.5 / 0.25 / 0.25 means 50% train, 25% val, 25% test.
    'train_split': 0.5,
    'val_split': 0.25,
    'seed': 42,
    'split_seed': 42,

    # Training config
    'num_epochs': 3,
    'learning_rate': 0.0001,
    'weight_decay': 1e-5,
    # Torchvision official model config
    # Supported names in the current environment:
    # resnet18, resnet34, resnet50, resnet101, resnet152,
    # resnext50_32x4d, resnext101_32x8d,
    # wide_resnet50_2, wide_resnet101_2
    #
    # Even though this dict is still called ResNet18_config for compatibility,
    # it now controls the full torchvision ResNet family.
    'model_name': 'resnet18',
    'use_pretrained': False,

    # Input/stem config，使用18，34，50
    # img_channels=3 for RGB images, 1 for grayscale images.
    'img_channels': 3,
    'first_conv_kernel_size': 7,
    'first_conv_stride': 2,
    'first_conv_padding': 3,
    'use_maxpool': True,
    'maxpool_kernel_size': 3,
    'maxpool_stride': 2,
    'maxpool_padding': 1,



    # Outputs
    'checkpoint_dir': os.path.join(PROJECT_ROOT, "checkpoint", "resnet18"),
}
