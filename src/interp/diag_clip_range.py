"""One-off diagnostic: is MindEye2's prior_out closer to clip_target computed from [0,1] images (as Train.ipynb does, which the
embedder turns into [0.5,1] via its own (x+1)/2) or from 2x-1 images (embedder then sees a proper [0,1])?
Uses the first N rows of evals/all_images.pt, which are the first N unique test images (same order as image_idx)."""
import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
sys.path.append('generative_models/')
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

p = argparse.ArgumentParser()
p.add_argument("--data_path", required=True)
p.add_argument("--bigg_safetensors", required=True)
p.add_argument("--prior_out", nargs="+", required=True, help="name=path pairs of saved (N,256,1664) prior_out files")
p.add_argument("--n", type=int, default=10)
args = p.parse_args()
device = torch.device("cuda")
print("device:", torch.cuda.get_device_name(0), flush=True)

import open_clip.factory as _f
from safetensors.torch import load_file
_orig = _f.load_state_dict
_f.load_state_dict = lambda path, *a, **k: load_file(path, device="cpu") if str(path).endswith(".safetensors") else _orig(path, *a, **k)

emb = FrozenOpenCLIPImageEmbedder(arch="ViT-bigG-14", version=args.bigg_safetensors, output_tokens=True, only_tokens=True).to(device)
imgs = torch.load(f"{args.data_path}/evals/all_images.pt", map_location="cpu", weights_only=False)[:args.n].float()
print("images range", imgs.min().item(), imgs.max().item())

def embed(x):
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        return emb(x.to(device)).float().cpu()

targets = {"as_trained [0,1]->[.5,1]": embed(imgs), "2x-1 -> proper [0,1]": embed(2 * imgs - 1)}
for k, t in targets.items():
    print(f"{k}: mean {t.mean():.4f} std {t.std():.4f} per-token norm {t.norm(dim=-1).mean():.2f}")
a, b = list(targets.values())
print("cos(clip_target_as_trained, clip_target_2x-1) per image:", [round(c, 3) for c in F.cosine_similarity(a.flatten(1), b.flatten(1)).tolist()])

n_ = lambda x: F.normalize(x.flatten(1), dim=-1)
for pair in args.prior_out:
    name, path = pair.split("=")
    po = torch.load(path, map_location="cpu", weights_only=False)[:args.n].float()
    print(f"\n== prior_out: {name}  (per-token norm {po.norm(dim=-1).mean():.2f})")
    for k, t in targets.items():
        cos = F.cosine_similarity(po.flatten(1), t.flatten(1))
        tok = F.cosine_similarity(po, t, dim=-1).mean(dim=1)
        rel = ((po - t) ** 2).flatten(1).sum(1) / (t ** 2).flatten(1).sum(1)
        M = n_(po) @ n_(t).T
        print(f"  vs {k:26s} flat cos mean {cos.mean():.3f} (min {cos.min():.3f}) | token cos {tok.mean():.3f} | rel SSE {rel.mean():.3f} | "
              f"top1 retrieval {(M.argmax(1) == torch.arange(len(po))).float().mean():.2f}")
