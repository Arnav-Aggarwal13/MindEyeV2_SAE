"""How close is the base run's prior_out to clip_target computed two ways, over all 1000 test images (CPU only)?
  as_trained: images in [0,1] fed straight in (what Train.ipynb does); the embedder's (x+1)/2 makes them [0.5,1]
  pm1:        2x-1 images, so the embedder sees a proper [0,1]
Paired per-image differences (as_trained minus pm1) with bootstrap 95% CIs."""
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--base_dir", required=True)
p.add_argument("--name", default="base")
args = p.parse_args()
D = f"{args.base_dir}/{args.name}_"
po = torch.load(D + "all_prior_out.pt")
targets = {"as_trained": torch.load(D + "clip_target.pt"), "pm1": torch.load(D + "clip_target_pm1.pt")}
N = len(po)
assert all(t.shape == po.shape for t in targets.values()) and N == 1000


def chunked(fn, *xs, bs=100):
    return torch.cat([fn(*[x[i:i + bs].float() for x in xs]) for i in range(0, N, bs)])


flat_cos = lambda a, b: F.cosine_similarity(a.flatten(1), b.flatten(1))
tok_cos = lambda a, b: F.cosine_similarity(a, b, dim=-1).mean(1)
rel_sse = lambda a, b: ((a - b) ** 2).flatten(1).sum(1) / (b ** 2).flatten(1).sum(1)
norm = lambda x: torch.cat([F.normalize(x[i:i + 100].flatten(1).float(), dim=-1) for i in range(0, N, 100)])

A = norm(po)
res, per = {}, {}
for name, t in targets.items():
    T = norm(t)
    M = A @ T.T
    ranks = (M > M.diag()[:, None]).sum(1)  # 0 = correct image ranked first
    per[name] = {"flat_cos": chunked(flat_cos, po, t), "token_cos": chunked(tok_cos, po, t), "rel_sse": chunked(rel_sse, po, t)}
    res[name] = {"flat_cos": per[name]["flat_cos"].mean().item(), "token_cos": per[name]["token_cos"].mean().item(),
                 "rel_sse": per[name]["rel_sse"].mean().item(), "top1_1000way": (ranks == 0).float().mean().item(),
                 "top5_1000way": (ranks < 5).float().mean().item(), "mean_rank": ranks.float().mean().item(),
                 "mean_offdiag_cos": ((M.sum() - M.diag().sum()) / (N * (N - 1))).item(), "per_token_norm": t.float().norm(dim=-1).mean().item()}
    del M, T
res["cos_between_targets_flat"] = flat_cos(targets["as_trained"].float(), targets["pm1"].float()).mean().item()

rng = np.random.default_rng(42)
paired = {}
for k in ["flat_cos", "token_cos", "rel_sse"]:
    d = (per["as_trained"][k] - per["pm1"][k]).numpy()
    boots = np.array([d[rng.integers(0, N, N)].mean() for _ in range(10000)])
    paired[k] = {"mean_diff_as_trained_minus_pm1": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                 "frac_images_as_trained_better": float((d > 0).mean() if k != "rel_sse" else (d < 0).mean())}
res["paired"] = paired

print(json.dumps(res, indent=1))
json.dump(res, open(f"{args.base_dir}/{args.name}_clip_target_variant_comparison.json", "w"), indent=1)
