"""Embedding-level metrics between brain-side tensors (prior_out, clip_voxels, later sae(prior_out)) and image-side tensors (clip_target,
later sae(clip_target)), all (N, 256, 1664), rows in the same image order. CPU only, no diffusion.

For every (brain, image) pair, in two variants ("raw" = upstream protocol, "centered" = each side minus its own per-token mean over the images):
  similarity: per-image flat cosine, mean per-token cosine, MSE, normalized error, norm ratio
  retrieval, S[i, j] = cosine(brain_i, image_j) of the flattened, L2-normalized embeddings, correct match on the diagonal:
    brain_to_image: for brain embedding i, rank of image i among all images   (argmax over each ROW of S)
    image_to_brain: for image j, rank of brain embedding j among all brain embeddings   (argmax over each COLUMN of S)
    - all-N version: top-1, top-5, mean rank, MRR from the per-item ranks, bootstrap 95% CI over images
    - upstream protocol: 30 random draws of 300 images, top-1 per draw, mean and normal 95% CI over the 30 draws (CI is optimistic: draws overlap)
With --ref, paired per-image differences (other minus ref brain) are also reported with bootstrap 95% CIs."""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--brain", nargs="+", required=True, help="name=path pairs")
p.add_argument("--image", nargs="+", required=True, help="name=path pairs")
p.add_argument("--ref", default=None, help="reference brain name for paired differences")
p.add_argument("--out_dir", required=True)
p.add_argument("--tag", default="embed_metrics")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
os.makedirs(args.out_dir, exist_ok=True)


def ranks_from_S(S):
    """S[i, j] = cos(brain_i, image_j). 0 = correct item ranked first."""
    d = S.diag()
    return (S > d[:, None]).sum(1), (S > d[None, :]).sum(0)  # (brain_to_image per brain row, image_to_brain per image column)


# self-check of the row/column convention on a tiny asymmetric example
_S = torch.tensor([[0.9, 0.1, 0.6], [0.8, 0.5, 0.1], [0.1, 0.2, 0.3]])
_b2i, _i2b = ranks_from_S(_S)
# row 1: image 0 (0.8) beats its own image 1 (0.5) -> rank 1; column 2: brain 0 (0.6) beats its own brain 2 (0.3) -> rank 1
assert _b2i.tolist() == [0, 1, 0] and _i2b.tolist() == [0, 0, 1], (_b2i, _i2b)

load = lambda spec: {s.split("=")[0]: torch.load(s.split("=")[1], map_location="cpu") for s in spec}
brains, images = load(args.brain), load(args.image)
N = len(next(iter(brains.values())))
for k, v in {**brains, **images}.items():
    assert v.shape == (N, 256, 1664), (k, v.shape)
print(f"N = {N}; brain: {list(brains)}; image: {list(images)}", flush=True)

rng = np.random.default_rng(args.seed)
draws = [rng.choice(N, size=300, replace=False) for _ in range(30)]  # same subsets for every pair
boot_idx = rng.integers(0, N, size=(2000, N))
ci = lambda per_item: [float(np.percentile(per_item[boot_idx].mean(1), q)) for q in (2.5, 97.5)]


def flat_norm(x, center):
    x = x.float()
    if center:
        x = x - x.mean(0, keepdim=True)
    return F.normalize(x.flatten(1), dim=-1)


def chunked(fn, a, b, bs=100):
    return torch.cat([fn(a[i:i + bs].float(), b[i:i + bs].float()) for i in range(0, N, bs)])


def similarity(b, t):
    flat_cos = chunked(lambda x, y: F.cosine_similarity(x.flatten(1), y.flatten(1)), b, t)
    tok_cos = chunked(lambda x, y: F.cosine_similarity(x, y, dim=-1).mean(1), b, t)
    rel_sse = chunked(lambda x, y: ((x - y) ** 2).flatten(1).sum(1) / (y ** 2).flatten(1).sum(1), b, t)
    sse = chunked(lambda x, y: ((x - y) ** 2).flatten(1).sum(1), b, t).sum()
    tvar = ((t.float() - t.float().mean(0, keepdim=True)) ** 2).sum()  # variance of the target around its per-token mean over images
    return {"flat_cos": flat_cos, "token_cos": tok_cos, "rel_sse": rel_sse}, {
        "flat_cos_mean": flat_cos.mean().item(), "token_cos_mean": tok_cos.mean().item(), "rel_sse_mean": rel_sse.mean().item(),
        "mse": (sse / b.numel()).item(), "nmse_vs_target_variance": (sse / tvar).item(),
        "norm_ratio_brain_over_image": (b.float().norm(dim=-1).mean() / t.float().norm(dim=-1).mean()).item()}


def retrieval(S):
    out, items = {}, {}
    rb, ri = ranks_from_S(S)
    for name, r in [("brain_to_image", rb), ("image_to_brain", ri)]:
        r = r.numpy()
        top1, top5, mrr = (r == 0).astype(float), (r < 5).astype(float), 1.0 / (r + 1)
        out[f"{name}_all{N}"] = {"top1": top1.mean(), "top1_ci95": ci(top1), "top5": top5.mean(), "top5_ci95": ci(top5),
                                 "mean_rank_1based": float((r + 1).mean()), "mean_rank_ci95": [x + 1 for x in ci(r.astype(float))], "mrr": mrr.mean()}
        items[f"{name}_rank"] = r
        sub = np.array([((S[np.ix_(idx, idx)].argmax(1 if name == "brain_to_image" else 0)).numpy() == np.arange(300)).mean() for idx in draws])
        se = sub.std() / np.sqrt(len(sub))
        out[f"{name}_upstream300x30"] = {"top1": sub.mean(), "ci95": [sub.mean() - 1.96 * se, sub.mean() + 1.96 * se], "chance": 1 / 300}
    return out, items


results, per_item, sims = {}, {}, {}
for bn, b in brains.items():
    for im, t in images.items():
        for variant in ["raw", "centered"]:
            B, T = flat_norm(b, variant == "centered"), flat_norm(t, variant == "centered")
            S = B @ T.T
            key = f"{bn}__vs__{im}__{variant}"
            if variant == "raw":
                sim_items, sim_summary = similarity(b, t)
            else:  # similarity of the centered tensors
                bc, tc = b.float() - b.float().mean(0, keepdim=True), t.float() - t.float().mean(0, keepdim=True)
                sim_items, sim_summary = similarity(bc, tc)
                del bc, tc
            r_summary, r_items = retrieval(S)
            results[key] = {**sim_summary, **r_summary,
                            "mean_cos_to_nonmatching": ((S.sum() - S.diag().sum()) / (N * (N - 1))).item()}
            per_item[key] = {**{k: v.numpy() for k, v in sim_items.items()}, **r_items}
            del B, T, S
            print(f"done {key}", flush=True)

# paired differences against a reference brain tensor (same image side and variant)
if args.ref:
    results["paired_vs_" + args.ref] = {}
    for key, items in per_item.items():
        bn, im, variant = key.split("__vs__")[0], key.split("__vs__")[1].split("__")[0], key.split("__")[-1]
        if bn == args.ref:
            continue
        ref = per_item[f"{args.ref}__vs__{im}__{variant}"]
        row = {}
        for m, hib in [("flat_cos", True), ("token_cos", True), ("rel_sse", False)]:
            d = items[m] - ref[m]
            row[m] = {"mean_diff": float(d.mean()), "ci95": ci(d)}
        for m in ["brain_to_image_rank", "image_to_brain_rank"]:
            d = (items[m] == 0).astype(float) - (ref[m] == 0).astype(float)
            row[m.replace("_rank", "_top1")] = {"mean_diff": float(d.mean()), "ci95": ci(d)}
            d = items[m].astype(float) - ref[m].astype(float)
            row[m.replace("_rank", "_mean_rank")] = {"mean_diff": float(d.mean()), "ci95": ci(d)}
        results["paired_vs_" + args.ref][key] = row

json.dump(results, open(f"{args.out_dir}/{args.tag}.json", "w"), indent=1, default=float)
np.savez(f"{args.out_dir}/{args.tag}_per_item.npz", **{f"{k}__{m}": v for k, d in per_item.items() for m, v in d.items()})

# readable table
print("\n" + "=" * 100)
for key, r in results.items():
    if key.startswith("paired_vs_"):
        continue
    print(f"\n{key}")
    print(f"  cosine: flat {r['flat_cos_mean']:.4f} | token {r['token_cos_mean']:.4f} | non-matching {r['mean_cos_to_nonmatching']:.4f} | "
          f"rel SSE {r['rel_sse_mean']:.3f} | nmse(var) {r['nmse_vs_target_variance']:.3f} | norm ratio {r['norm_ratio_brain_over_image']:.3f}")
    for d in ["brain_to_image", "image_to_brain"]:
        a, u = r[f"{d}_all{N}"], r[f"{d}_upstream300x30"]
        print(f"  {d:15s} all-{N}: top1 {a['top1']:.4f} [{a['top1_ci95'][0]:.4f},{a['top1_ci95'][1]:.4f}] top5 {a['top5']:.4f} "
              f"mean rank {a['mean_rank_1based']:.2f} MRR {a['mrr']:.4f} | 300x30: top1 {u['top1']:.4f} [{u['ci95'][0]:.4f},{u['ci95'][1]:.4f}]")
