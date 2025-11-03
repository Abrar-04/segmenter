from segm.model.hydravit import HydraViT

class HydraViTEncoder(HydraViT):
    """
    Wrapper around HydraViT so it behaves like the VisionTransformer
    expected by Segmenter. The only purpose of this class is to
    maintain compatibility with Segmenter.forward().
    It preserves the 'return_features=True' interface and defines
    a dummy 'distilled' flag expected by the Segmenter.
    """

    def __init__(self, *args, **kwargs):
        # Rename Segmenter-style args to HydraViT-style
        rename_map = {
            "image_size": "img_size",
            "n_layers": "depth",
            "n_heads": "num_heads",
            "d_model": "embed_dim",
            "dropout": "drop_rate",
            "drop_path_rate": "drop_path_rate",
            "normalization": None,
            "distilled": None,
            "backbone": None,
            "n_cls": None,
        }

        remapped = {}
        for k, v in kwargs.items():
            if k in rename_map:
                newk = rename_map[k]
                if newk:         # keep and rename
                    remapped[newk] = v
                else:
                    continue     # drop keys mapped to None
            else:
                remapped[k] = v  # untouched


        # Handle number of classes separately

        self.n_cls = kwargs.get("n_cls", 1000)  # default same as ViT

        # Supply default values if not provided

        remapped.setdefault("img_size", (224, 224))
        remapped.setdefault("patch_size", 16)
        remapped.setdefault("in_chans", 3)
        remapped.setdefault("num_classes", self.n_cls)
        remapped.setdefault("global_pool", "token")
        remapped.setdefault("embed_dim", 768)
        remapped.setdefault("depth", 12)
        remapped.setdefault("num_heads", 12)
        remapped.setdefault("mlp_ratio", 4.0)
        remapped.setdefault("qkv_bias", True)
        remapped.setdefault("qk_norm", False)
        remapped.setdefault("init_values", None)
        remapped.setdefault("class_token", True)
        remapped.setdefault("no_embed_class", False)
        remapped.setdefault("pre_norm", False)
        remapped.setdefault("fc_norm", None)
        remapped.setdefault("dynamic_img_size", False)
        remapped.setdefault("dynamic_img_pad", False)
        remapped.setdefault("drop_rate", 0.0)
        remapped.setdefault("pos_drop_rate", 0.0)
        remapped.setdefault("patch_drop_rate", 0.0)
        remapped.setdefault("proj_drop_rate", 0.0)
        remapped.setdefault("attn_drop_rate", 0.0)
        remapped.setdefault("drop_path_rate", 0.0)
        remapped.setdefault("weight_init", "")
        # embed_layer, norm_layer, act_layer, block_fn, mlp_layer use HydraViT defaults


        print("⚙️ HydraViTEncoder final kwargs:", list(remapped.keys()))
        super().__init__(*args, **remapped)


        self.distilled = False
        self.d_model = getattr(self, "embed_dim", None)
        self.patch_size = getattr(self.patch_embed, "patch_size", None)
        
        # Normalize patch_size for Segmenter
        # HydraViT uses a tuple, Segmenter expects an int
        if isinstance(self.patch_size, (tuple, list)):
            self.patch_size = self.patch_size[0]

    def forward(self, im, return_features=False, p=None):
        """
        If return_features=True → return patch embeddings (for decoder input).
        Otherwise → return classification logits via HydraViT head.
        """

        # default p to full embed_dim if p not found, handle p later
        if p is None:
            p = getattr(self, "embed_dim", 768)

        x = self.forward_features(im, p)
        if return_features:
            return x
        return self.forward_head(x)
