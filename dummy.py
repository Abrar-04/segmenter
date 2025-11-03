import torch
from segm.model.factory import create_segmenter
from segm.model.utils import num_params

# === Minimal HydraViT + MaskTransformer sanity test ===
# mimic a typical config dictionary as Train.py would pass
net_kwargs = {
    "backbone": "hydravit_dyn_patch16_224",
    "image_size": (224, 224),
    "patch_size": 16,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4,
    "drop_path_rate": 0.0,
    "dropout": 0.1,
    "global_pool": "token",
    "class_token": True,
    "normalization": "vit",
    "decoder": {
        "name": "mask_transformer",
        "n_layers": 2,
        "drop_path_rate": 0.0,
        "dropout": 0.1,
    },
    "n_cls": 150,   # ADE20K
}

# create model (factory.py automatically builds HydraViT + decoder + Segmenter)
model = create_segmenter(net_kwargs)

print("✅ Model created successfully")
print(f"Encoder parameters: {num_params(model.encoder):,}")
print(f"Decoder parameters: {num_params(model.decoder):,}")

# run a dummy forward pass
x = torch.randn(1, 3, 224, 224)  # 1 image, 3 channels, 224×224
with torch.no_grad():
    y = model(x)

print(f"✅ Forward pass OK — output shape: {tuple(y.shape)}")
