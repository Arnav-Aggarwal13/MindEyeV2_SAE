"""Script port of the image-level and brain-space parts of src/final_evaluations.ipynb, run on saved reconstructions only.

Run from src/. Metrics (all per image over the 1000 unique test images, same protocol as upstream):
  PixCorr, SSIM, AlexNet(2), AlexNet(5), InceptionV3, CLIP (two-way identification), EffNet-B, SwAV (distances), and
  GNet brain correlation for nsd_general / V1 / V2 / V3 / V4 / higher_vis.
Sections (--sections, comma separated; default "image,brain"):
  image, brain   need --recons_path (and --betas_avg for brain); GPU
  captions       needs --predcaptions (*_all_predcaptions.pt): METEOR, ROUGE-1/L, MiniLM, CLIP-B/32, CLIP-L/14 text similarity
                 for GROUND (GIT-on-image vs human), ABSOLUTE (brain vs human, brain vs GIT-on-image) and RELATIVE (brain-vs-image / ground)
  umap           needs --clip_target plus --prior_out and/or --clipvoxels: UMAP of clip_target (image embeddings) vs prior_out
                 ("CLIP Image x Diffusion Prior") and vs clip_voxels ("CLIP Image x Retrieval Submodule"); CPU is enough
The upstream retrieval cell is covered by embed_metrics.py. The upstream UMAP of `backbone` is not run (the base run did not save it).
The upstream "enhanced" blending with blurry recons is intentionally absent: regular recons only. Authors' files only support image,brain.

Differences from the notebook:
  - `image_clip_embedder(all_images)` in the UMAP cells is the same tensor as clip_target; we load the saved clip_target instead
  - paths are arguments; averaged test betas come from the base run's base_test_betas_avg.pt (identical selection/averaging)
  - SSIM: skimage >= 0.21 removed `multichannel`. `SSIM` uses channel_axis=-1 (what multichannel=True did on the 2-D grayscale
    arrays, i.e. each image column is a "channel"); `SSIM_2d` is the standard 2-D SSIM. Both are reported.
  - torch.load(weights_only=False) (torch >= 2.6 default breaks the GNet checkpoint)
  - EffNet / SwAV features are extracted on the GPU in batches of 100
  - each metric runs in its own try/except so one failure does not lose the others; per-image values are saved to an .npz
"""
import os
import sys
import json
import time
import argparse
import traceback
import numpy as np
import h5py
import scipy as sp
import torch
from tqdm import tqdm
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
torch.backends.cuda.matmul.allow_tf32 = True
_orig_torch_load = torch.load
torch.load = lambda *a, **k: _orig_torch_load(*a, **{**k, "weights_only": False})

import utils
from models import GNet8_Encoder
from torchvision.models.feature_extraction import create_feature_extractor

parser = argparse.ArgumentParser()
parser.add_argument("--sections", default="image,brain", help="comma separated subset of: image, brain, captions, umap")
parser.add_argument("--tag", default="", help="suffix for the output files (metrics<_tag>.json), so several runs can share one out_name")
parser.add_argument("--recons_path", default=None, help="(1000,3,H,W) float tensor in [0,1], rows in np.unique(test image idx) order")
parser.add_argument("--betas_avg", default=None, help="(1000, n_vox) fp32 averaged test betas, same row order as the recons")
parser.add_argument("--predcaptions", default=None, help="*_all_predcaptions.pt for the captions section")
parser.add_argument("--clip_target", default=None, help="(1000,256,1664) clip_target for the umap section")
parser.add_argument("--prior_out", default=None, help="(1000,256,1664) prior_out for the umap section")
parser.add_argument("--clipvoxels", default=None, help="(1000,256,1664) clip_voxels for the umap section")
parser.add_argument("--out_name", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--data_path", required=True, help="flat dir with evals/all_images.pt, all_captions.pt, brain_region_masks.hdf5, gnet_multisubject.pt")
parser.add_argument("--cache_dir", default=None)
parser.add_argument("--subj", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
cache_dir = args.cache_dir or args.data_path
subj = args.subj
sections = set(args.sections.split(","))
assert sections <= {"image", "brain", "captions", "umap"}, sections
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "| sections:", sorted(sections), flush=True)
if sections & {"image", "brain"}:
    assert device.type == "cuda" and args.recons_path
if "brain" in sections:
    assert args.betas_avg
utils.seed_everything(args.seed)
out_dir = os.path.join(args.out_dir, args.out_name)
os.makedirs(out_dir, exist_ok=True)

# ---- data ----
n_images = 1000
if sections & {"image", "brain"}:
    all_images = torch.load(f"{args.data_path}/evals/all_images.pt", map_location="cpu")
    all_recons = torch.load(args.recons_path, map_location="cpu")
    print("all_images", tuple(all_images.shape), "all_recons", tuple(all_recons.shape), all_recons.dtype,
          f"recon range [{all_recons.min():.3f}, {all_recons.max():.3f}]", flush=True)
    assert len(all_images) == len(all_recons) == n_images
    imsize = 256
    if all_images.shape[-1] != imsize:
        all_images = transforms.Resize((imsize, imsize))(all_images).float()
    if all_recons.shape[-1] != imsize:
        all_recons = transforms.Resize((imsize, imsize))(all_recons).float()

results, per_image, errors = {}, {}, {}


def run(name, fn):
    t = time.time()
    try:
        fn()
        print(f"[{time.time() - t:6.1f}s] done {name}", flush=True)
    except Exception:
        errors[name] = traceback.format_exc()
        print(f"[{time.time() - t:6.1f}s] FAILED {name}\n{errors[name]}", flush=True)


@torch.no_grad()
def two_way_identification(all_recons, all_images, model, preprocess, feature_layer=None):
    preds = model(torch.stack([preprocess(recon) for recon in all_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()
    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)
    success_cnt = np.sum(r < congruents, 0)  # per recon: number of distractors that correlate less than the true image
    return success_cnt / (len(all_images) - 1)


# ---- PixCorr ----
def pixcorr():
    pre = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    imgs_f = pre(all_images).reshape(len(all_images), -1).cpu()
    recs_f = pre(all_recons).view(len(all_recons), -1).cpu()
    vals = np.array([np.corrcoef(imgs_f[i], recs_f[i])[0][1] for i in tqdm(range(len(imgs_f)))])
    per_image["PixCorr"], results["PixCorr"] = vals, float(vals.mean())


# ---- SSIM ----
def ssim_metric():
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity as ssim
    pre = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    img_gray = rgb2gray(pre(all_images).permute((0, 2, 3, 1)).cpu().numpy())
    recon_gray = rgb2gray(pre(all_recons).permute((0, 2, 3, 1)).cpu().numpy())
    kw = dict(gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
    for name, ch in [("SSIM_2d", None), ("SSIM", -1)]:
        try:
            vals = np.array([ssim(rec, im, channel_axis=ch, **kw) for im, rec in tqdm(zip(img_gray, recon_gray), total=len(img_gray), desc=name)])
            per_image[name], results[name] = vals, float(vals.mean())
        except Exception:
            errors[name] = traceback.format_exc()
            print(f"FAILED {name}\n{errors[name]}", flush=True)


# ---- AlexNet ----
def alexnet_metrics():
    from torchvision.models import alexnet, AlexNet_Weights
    alex_model = create_feature_extractor(alexnet(weights=AlexNet_Weights.IMAGENET1K_V1), return_nodes=['features.4', 'features.11']).to(device)
    alex_model.eval().requires_grad_(False)
    pre = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    for name, layer in [("AlexNet(2)", 'features.4'), ("AlexNet(5)", 'features.11')]:
        v = two_way_identification(all_recons.to(device).float(), all_images, alex_model, pre, layer)
        per_image[name], results[name] = v, float(v.mean())
        print(f"{name} 2-way: {results[name]:.4f}", flush=True)


# ---- InceptionV3 ----
def inception_metric():
    from torchvision.models import inception_v3, Inception_V3_Weights
    m = create_feature_extractor(inception_v3(weights=Inception_V3_Weights.DEFAULT), return_nodes=['avgpool']).to(device)
    m.eval().requires_grad_(False)
    pre = transforms.Compose([transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    v = two_way_identification(all_recons, all_images, m, pre, 'avgpool')
    per_image["InceptionV3"], results["InceptionV3"] = v, float(v.mean())


# ---- CLIP ViT-L/14 ----
def clip_metric():
    import clip
    clip_model, _ = clip.load("ViT-L/14", device=device, download_root=f"{os.environ.get('TORCH_HOME', os.path.expanduser('~/.cache/torch'))}/clip")
    pre = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                              transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    v = two_way_identification(all_recons, all_images, clip_model.encode_image, pre, None)
    per_image["CLIP"], results["CLIP"] = v, float(v.mean())


# ---- EffNet-B / SwAV (1 - Pearson correlation of pooled features; lower is better) ----
@torch.no_grad()
def pooled_feats(model, pre, imgs, bs=100):
    out = []
    for i in range(0, len(imgs), bs):
        out.append(model(pre(imgs[i:i + bs]).to(device))['avgpool'].flatten(1).float().cpu())
    return torch.cat(out).numpy()


def distance_metric(name, model, pre):
    model = create_feature_extractor(model, return_nodes=['avgpool']).to(device)
    model.eval().requires_grad_(False)
    gt, fake = pooled_feats(model, pre, all_images), pooled_feats(model, pre, all_recons)
    d = np.array([sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))])
    per_image[name], results[name] = d, float(d.mean())


def effnet_metric():
    from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
    pre = transforms.Compose([transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    distance_metric("EffNet-B", efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT), pre)


def swav_metric():
    pre = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    distance_metric("SwAV", torch.hub.load('facebookresearch/swav:main', 'resnet50'), pre)


# ---- brain correlation (GNet predictions for each recon vs measured, 3-rep-averaged betas) ----
def brain_metric():
    from torchmetrics import PearsonCorrCoef
    with h5py.File(f"{args.data_path}/brain_region_masks.hdf5", "r") as f:
        g = f[f"subj0{subj}"]
        masks = {r: g[r][:] for r in ["nsd_general", "V1", "V2", "V3", "V4", "higher_vis"]}
    betas = torch.load(args.betas_avg, map_location="cpu").float()
    assert betas.shape[0] == len(all_recons), betas.shape
    recon_list = [transforms.ToPILImage()(all_recons[i].detach()) for i in range(len(all_recons))]
    GNet = GNet8_Encoder(device=device, subject=subj, model_path=f"{cache_dir}/gnet_multisubject.pt")
    PeC = PearsonCorrCoef(num_outputs=len(recon_list))
    beta_primes = GNet.predict(recon_list)
    for region, mask in masks.items():
        score = PeC(betas[:, mask].moveaxis(0, 1), beta_primes[:, mask].moveaxis(0, 1))
        per_image[f"BrainCorr {region}"] = score.cpu().numpy()
        results[f"BrainCorr {region}"] = float(torch.mean(score))
        print(f"BrainCorr {region}: {results[f'BrainCorr {region}']:.4f}", flush=True)


# ---- captions: three caption sets (human COCO, GIT on the real image, brain-derived prior_out -> GIT) ----
def caption_metrics():
    import evaluate
    from sentence_transformers import SentenceTransformer, util
    from transformers import CLIPModel, AutoTokenizer
    as_list = lambda x: [str(c) for c in x]
    all_captions = as_list(torch.load(f"{args.data_path}/evals/all_captions.pt"))
    all_git = as_list(torch.load(f"{args.data_path}/evals/all_git_generated_captions.pt"))
    all_pred = as_list(torch.load(args.predcaptions))
    assert len(all_captions) == len(all_git) == len(all_pred) == n_images, (len(all_captions), len(all_git), len(all_pred))
    print("example captions:", all_captions[0], "|", all_git[0], "|", all_pred[0], flush=True)
    cap = {}

    # METEOR / ROUGE: corpus-level means (as upstream; no per-image values)
    meteor = evaluate.load('meteor')
    m_img_ref = meteor.compute(predictions=all_git, references=all_captions)["meteor"]
    m_brain_ref = meteor.compute(predictions=all_pred, references=all_captions)["meteor"]
    m_brain_img = meteor.compute(predictions=all_pred, references=all_git)["meteor"]
    cap["Meteor_img_ref"], cap["Meteor_brain_ref"], cap["Meteor_brain_img"], cap["Meteor_relative"] = m_img_ref, m_brain_ref, m_brain_img, m_brain_img / m_img_ref
    rouge = evaluate.load('rouge')
    r_img_ref = rouge.compute(predictions=all_git, references=all_captions)
    r_brain_ref = rouge.compute(predictions=all_pred, references=all_captions)
    r_brain_img = rouge.compute(predictions=all_pred, references=all_git)
    for key, short in [("rouge1", "Rouge1"), ("rougeL", "RougeL")]:
        cap[f"{short}_img_ref"], cap[f"{short}_brain_ref"], cap[f"{short}_brain_img"] = r_img_ref[key], r_brain_ref[key], r_brain_img[key]
        cap[f"{short}_relative"] = r_brain_img[key] / r_img_ref[key]

    def add_similarity(prefix, emb_brain, emb_cap, emb_img):
        sims = {"brain_img": util.pytorch_cos_sim(emb_brain, emb_img).cpu().diag(), "brain_ref": util.pytorch_cos_sim(emb_brain, emb_cap).cpu().diag(),
                "img_ref": util.pytorch_cos_sim(emb_img, emb_cap).cpu().diag()}
        for k, v in sims.items():
            cap[f"{prefix}_{k}"] = v.mean().item()
            per_image[f"Caption/{prefix}_{k}"] = v.numpy()
        cap[f"{prefix}_relative"] = (sims["brain_img"].mean() / sims["img_ref"].mean()).item()

    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    with torch.no_grad():
        enc = lambda x: sentence_model.encode(x, convert_to_tensor=True)
        add_similarity("Sentence", enc(all_pred), enc(all_captions), enc(all_git))
    for prefix, hf_name in [("CLIP-B", "openai/clip-vit-base-patch32"), ("CLIP-L", "openai/clip-vit-large-patch14")]:
        model_clip, tokenizer = CLIPModel.from_pretrained(hf_name), AutoTokenizer.from_pretrained(hf_name)
        with torch.no_grad():
            enc = lambda x: model_clip.get_text_features(**tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=77))
            add_similarity(prefix, enc(all_pred), enc(all_captions), enc(all_git))
    for k, v in cap.items():
        results[f"Caption/{k}"] = float(v)
        print(f"Caption/{k}: {v:.4f}", flush=True)


# ---- UMAP: clip_target (image embeddings, blue) vs a brain-side embedding, flattened ----
def umap_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    clip_target = torch.load(args.clip_target).flatten(1).float().numpy()
    reducer = umap.UMAP(random_state=args.seed)
    jobs = []
    if args.prior_out:
        jobs.append(("prior_out", args.prior_out, "Prior_Out", "red", "CLIP Image x Diffusion Prior", 1.0))
    if args.clipvoxels:
        jobs.append(("clipvoxels", args.clipvoxels, "Voxels", "orange", "CLIP Image x Retrieval Submodule", .5))
    for name, path, label, color, title, alpha in jobs:
        other = torch.load(path).flatten(1).float().numpy()
        assert other.shape == clip_target.shape, (other.shape, clip_target.shape)
        embedding = reducer.fit_transform(np.concatenate((clip_target, other), axis=0))
        euclidean_dist = float(np.mean(np.linalg.norm(embedding[:len(clip_target), :] - embedding[len(clip_target):, :], axis=1)))
        plt.figure(figsize=(5, 5))
        plt.scatter(embedding[:len(clip_target), 0], embedding[:len(clip_target), 1], c='blue', label='Images', alpha=alpha if name == "clipvoxels" else 1.0)
        plt.scatter(embedding[len(clip_target):, 0], embedding[len(clip_target):, 1], c=color, label=label, alpha=alpha if name == "clipvoxels" else 1.0)
        plt.title(f'{title}\nAvg. euclidean distance = {euclidean_dist:.2f}')
        plt.legend()
        plt.savefig(f'{out_dir}/umap_{name}.png', dpi=150, bbox_inches="tight")
        plt.close()
        np.save(f"{out_dir}/umap_{name}_coords.npy", embedding)
        results[f"UMAP avg euclidean distance {name}"] = euclidean_dist
        print(f"UMAP {name}: avg euclidean distance {euclidean_dist:.3f}", flush=True)


plan = []
if "image" in sections:
    plan += [("PixCorr", pixcorr), ("SSIM", ssim_metric), ("AlexNet", alexnet_metrics), ("InceptionV3", inception_metric),
             ("CLIP", clip_metric), ("EffNet-B", effnet_metric), ("SwAV", swav_metric)]
if "brain" in sections:
    plan += [("BrainCorr", brain_metric)]
if "captions" in sections:
    assert args.predcaptions
    plan += [("Captions", caption_metrics)]
if "umap" in sections:
    assert args.clip_target and (args.prior_out or args.clipvoxels)
    plan += [("UMAP", umap_plots)]
for name, fn in plan:
    run(name, fn)
    torch.cuda.empty_cache()

# ---- save ----
print("\n" + "=" * 50 + f"\n{args.out_name} {sorted(sections)}")
for k, v in results.items():
    print(f"{k:40s} {v:.4f}")
if errors:
    print("FAILED metrics:", list(errors))
suffix = f"_{args.tag}" if args.tag else ""
with open(f"{out_dir}/metrics{suffix}.json", "w") as f:
    json.dump({"sections": sorted(sections), "recons_path": args.recons_path, "predcaptions": args.predcaptions, "n_images": n_images,
               "results": results, "errors": errors}, f, indent=1)
if per_image:
    np.savez(f"{out_dir}/per_image{suffix}.npz", **{k.replace(" ", "_").replace("/", "_"): v for k, v in per_image.items()})
with open(f"{out_dir}/metrics{suffix}.csv", "w") as f:
    f.write("Metric\tValue\n" + "".join(f"{k}\t{v:.6f}\n" for k, v in results.items()))
sys.exit(1 if errors else 0)
