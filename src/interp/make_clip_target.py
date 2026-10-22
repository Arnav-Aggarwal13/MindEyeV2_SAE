"""Compute clip_target for the 1000 test images: the OpenCLIP bigG token embedding MindEye2's diffusion prior is trained to predict.

Replicates Train.ipynb exactly: FrozenOpenCLIPImageEmbedder(arch="ViT-bigG-14", output_tokens=True, only_tokens=True) called on the
224x224 float32 images (values in [0, 1], no augmentation) under torch.cuda.amp.autocast(float16), no extra normalization.
Note the embedder's own preprocess() computes (x + 1) / 2 and then CLIP mean/std normalization, which Train also applies to [0, 1]
images; we keep that so clip_target is the same tensor the prior was trained against.

Run from src/. Row i is evals/all_images.pt row i, which recon_infer.py verifies is np.unique(test_images_idx) order (same as image_idx).
"""
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
sys.path.append('generative_models/')
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="flat dir containing evals/all_images.pt")
parser.add_argument("--bigg_safetensors", type=str, required=True, help="local open_clip bigG-14 laion2b_s39b_b160k safetensors")
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--out_name", type=str, default="base")
parser.add_argument("--batch_size", type=int, default=50)
args = parser.parse_args()

device = torch.device("cuda")
print("device:", torch.cuda.get_device_name(0), flush=True)
out_dir = os.path.join(args.out_dir, args.out_name)
os.makedirs(out_dir, exist_ok=True)

all_images = torch.load(f"{args.data_path}/evals/all_images.pt", map_location="cpu", weights_only=False)
print("all_images", tuple(all_images.shape), all_images.dtype, f"range [{all_images.min():.3f}, {all_images.max():.3f}]")
assert all_images.ndim == 4 and all_images.shape[1:] == (3, 224, 224)

# open_clip 2.24 cannot read .safetensors; let it (same shim as recon_infer.py)
import open_clip.factory as _oc_factory
from safetensors.torch import load_file as _load_safetensors
_orig_load_state_dict = _oc_factory.load_state_dict


def _load_state_dict(checkpoint_path, *a, **k):
    if str(checkpoint_path).endswith(".safetensors"):
        return _load_safetensors(checkpoint_path, device="cpu")
    return _orig_load_state_dict(checkpoint_path, *a, **k)


_oc_factory.load_state_dict = _load_state_dict

clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version=args.bigg_safetensors,
    output_tokens=True,
    only_tokens=True,
)
clip_img_embedder.to(device)

out = []
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
    for i in tqdm(range(0, len(all_images), args.batch_size)):
        clip_target = clip_img_embedder(all_images[i:i + args.batch_size].to(device))
        assert not torch.any(torch.isnan(clip_target))
        if i == 0:
            print("embedder output dtype:", clip_target.dtype, "shape:", tuple(clip_target.shape), flush=True)
        out.append(clip_target.half().cpu())
clip_target = torch.vstack(out)
assert clip_target.shape == (len(all_images), 256, 1664), clip_target.shape
assert torch.isfinite(clip_target.float()).all()  # fp16 cast must not overflow

print(f"clip_target {tuple(clip_target.shape)} {clip_target.dtype}: mean {clip_target.float().mean():.4f} std {clip_target.float().std():.4f} "
      f"per-token L2 norm mean {clip_target.float().norm(dim=-1).mean():.2f}")
torch.save(clip_target, f"{out_dir}/{args.out_name}_clip_target.pt")
print(f"saved {out_dir}/{args.out_name}_clip_target.pt")
