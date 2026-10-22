"""Script port of src/recon_inference.ipynb (regular unCLIP recons only, no enhancement).

Run from src/. Differences from the notebook, all needed to run on this cluster:
  - paths are arguments (no /weka paths); --hidden_dim defaults to 4096 for the released checkpoint
  - `plotting` is not used; no bare `err`
  - SDXL unCLIP uses PyTorch attention (the installed xformers has no CUDA kernels)
  - torch.load(weights_only=False) for the trusted checkpoints (torch >= 2.6 defaults to True)
  - the bigG weights load from a local safetensors file (open_clip 2.24 only reads .bin) so no download is needed
  - the checkpoint load is strict and fails loudly (no bare except / deepspeed fallback)
  - noise is seeded per image so a rerun gives identical outputs
  - --max_images limits the run to the first N unique test images (smoke test)
  - outputs are written in shards of --shard_size images; --resume skips finished shards (preemptable partition)
  - also saves the 3-rep-averaged test betas, a run record, and an image-order check against all_images.pt
"""
import os
import sys
import glob
import json
import time
import argparse
import datetime
import subprocess
import importlib.metadata as md
import numpy as np
import h5py
from tqdm import tqdm
import webdataset as wds

import torch
import torch.nn as nn
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.models.diffusion import DiffusionEngine
from omegaconf import OmegaConf

torch.backends.cuda.matmul.allow_tf32 = True

import utils
from models import BrainNetwork, PriorNetwork, BrainDiffusionPrior

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="final_subj01_pretrained_40sess_24bs")
parser.add_argument("--data_path", type=str, required=True, help="flat dir with betas, wds/, sd_image_var_autoenc.pth, ...")
parser.add_argument("--cache_dir", type=str, default=None, help="dir with bigG_to_L, unclip6 ckpt, autoenc ckpt (default: data_path)")
parser.add_argument("--ckpt_root", type=str, default=None, help="dir containing <model_name>/last.pth (default: data_path/train_logs)")
parser.add_argument("--bigg_safetensors", type=str, required=True, help="local open_clip bigG-14 laion2b_s39b_b160k safetensors")
parser.add_argument("--unclip_ckpt", type=str, default=None, help="default: cache_dir/unclip6_epoch0_step110000.ckpt")
parser.add_argument("--out_dir", type=str, default="evals")
parser.add_argument("--out_name", type=str, default=None, help="output subfolder name (default: model_name)")
parser.add_argument("--subj", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument("--blurry_recon", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--n_blocks", type=int, default=4)
parser.add_argument("--hidden_dim", type=int, default=4096)
parser.add_argument("--new_test", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_images", type=int, default=0, help="0 = all unique test images")
parser.add_argument("--shard_size", type=int, default=100)
parser.add_argument("--resume", action="store_true", help="skip shards already in <out_dir>/partial/")
args = parser.parse_args()

data_path = args.data_path
cache_dir = args.cache_dir or data_path
ckpt_root = args.ckpt_root or os.path.join(data_path, "train_logs")
out_name = args.out_name or args.model_name
subj, hidden_dim, seed = args.subj, args.hidden_dim, args.seed
assert "enhanced" not in out_name  # the eval notebook blends blurry recons into any name containing "enhanced"

device = torch.device("cuda")
print("device:", torch.cuda.get_device_name(0), flush=True)
utils.seed_everything(seed)
out_dir = os.path.join(args.out_dir, out_name)
os.makedirs(out_dir, exist_ok=True)


stage_times = {}


def stage(msg, t0=[time.time()]):
    now = time.time()
    print(f"[{now - t0[0]:7.1f}s] {msg}", flush=True)
    stage_times[msg] = round(now - t0[0], 1)
    t0[0] = now


# ---- data: betas + test trial indices ----
f = h5py.File(f'{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5', 'r')
betas = torch.Tensor(f['betas'][:]).to("cpu")
num_voxels = betas[0].shape[-1]
print(f"num_voxels for subj0{subj}: {num_voxels}")

if not args.new_test:
    num_test = {3: 2113, 4: 1985, 6: 2113, 8: 1985}.get(subj, 2770)
    test_url = f"{data_path}/wds/subj0{subj}/test/0.tar"
else:
    num_test = {3: 2371, 4: 2188, 6: 2371, 8: 2188}.get(subj, 3000)
    test_url = f"{data_path}/wds/subj0{subj}/new_test/0.tar"
print(test_url)


def my_split_by_node(urls): return urls


test_data = wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node)\
    .decode("torch")\
    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True)

test_images_idx = []
for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
    test_voxels = betas[behav[:, 0, 5].cpu().long()]
    test_images_idx = np.append(test_images_idx, behav[:, 0, 0].cpu().numpy())
test_images_idx = test_images_idx.astype(int)
assert (test_i + 1) * num_test == len(test_voxels) == len(test_images_idx)
del betas
uniq_all = np.unique(test_images_idx)
print(f"{len(test_voxels)} test trials, {len(uniq_all)} unique images")
stage("data loaded")

# ---- image-order check: evals/all_images.pt rows must be in np.unique(test_images_idx) order ----
if args.new_test:
    assert len(uniq_all) == 1000, len(uniq_all)
assert np.all(np.diff(uniq_all) > 0)  # sorted and unique
all_images = torch.load(f"{data_path}/evals/all_images.pt", map_location="cpu", weights_only=False)
assert len(all_images) == len(uniq_all), (all_images.shape, len(uniq_all))
check_rows = np.unique(np.linspace(0, len(uniq_all) - 1, 20).astype(int))
with h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r') as coco_f:
    coco_images = coco_f['images']
    row_diffs = []
    for i in check_rows:
        ref = torch.from_numpy(np.asarray(coco_images[int(uniq_all[i])])).float()
        if ref.shape != all_images[i].shape:  # tolerate HWC storage
            ref = ref.permute(2, 0, 1)
        assert ref.shape == all_images[i].shape, (ref.shape, all_images[i].shape)
        row_diffs.append(float((ref - all_images[i].float()).abs().max()))
order_ok = bool(max(row_diffs) < 1e-3)
order_check = {"all_images_shape": list(all_images.shape), "n_unique": int(len(uniq_all)), "sorted_unique": True,
               "rows_checked": check_rows.tolist(), "coco_idx_checked": uniq_all[check_rows].tolist(),
               "max_abs_diff_per_row": row_diffs, "tolerance": 1e-3, "passed": order_ok}
with open(f"{out_dir}/{out_name}_order_check.json", "w") as fj:
    json.dump(order_check, fj, indent=1)
assert order_ok, f"all_images.pt rows are NOT in np.unique order, max diffs {row_diffs}"
del all_images
stage("image-order check passed")

# ---- MindEye2 ----
clip_seq_dim, clip_emb_dim = 256, 1664

if args.blurry_recon:
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D'] * 4,
        up_block_types=['UpDecoderBlock2D'] * 4,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    autoenc.load_state_dict(torch.load(f'{cache_dir}/sd_image_var_autoenc.pth', map_location='cpu', weights_only=False))
    autoenc.eval().requires_grad_(False)
    autoenc.to(device)


class RidgeRegression(torch.nn.Module):
    def __init__(self, input_sizes, out_features):
        super().__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([torch.nn.Linear(s, out_features) for s in input_sizes])

    def forward(self, x, subj_idx):
        return self.linears[subj_idx](x[:, 0]).unsqueeze(1)


model = nn.Module()
model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)
model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1,
                              clip_size=clip_emb_dim, out_dim=clip_emb_dim * clip_seq_dim)
prior_network = PriorNetwork(dim=clip_emb_dim, depth=6, dim_head=52, heads=clip_emb_dim // 52,
                             causal=False, num_tokens=clip_seq_dim, learned_query_mode="pos_emb")
model.diffusion_prior = BrainDiffusionPrior(net=prior_network, image_embed_dim=clip_emb_dim,
                                            condition_on_text_encodings=False, timesteps=100,
                                            cond_drop_prob=0.2, image_embed_scale=None)

ckpt_file = f'{ckpt_root}/{args.model_name}/last.pth'
print(f"loading {ckpt_file}")
checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
del checkpoint
model.to(device).eval().requires_grad_(False)
stage("MindEye2 loaded")

# ---- captioning ----
from transformers import AutoProcessor
from modeling_git import GitForCausalLMClipEmb
processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
clip_text_model.to(device).eval().requires_grad_(False)
clip_text_seq_dim, clip_text_emb_dim = 257, 1024


class CLIPConverter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(clip_seq_dim, clip_text_seq_dim)
        self.linear2 = nn.Linear(clip_emb_dim, clip_text_emb_dim)

    def forward(self, x):
        x = self.linear1(x.permute(0, 2, 1))
        return self.linear2(x.permute(0, 2, 1))


clip_convert = CLIPConverter()
clip_convert.load_state_dict(torch.load(f"{cache_dir}/bigG_to_L_epoch8.pth", map_location='cpu', weights_only=False)['model_state_dict'], strict=True)
clip_convert.to(device)
stage("GIT + converter loaded")

# ---- unCLIP ----
# open_clip 2.24 cannot read .safetensors; let it
import open_clip.factory as _oc_factory
from safetensors.torch import load_file as _load_safetensors
_orig_load_state_dict = _oc_factory.load_state_dict


def _load_state_dict(checkpoint_path, *a, **k):
    if str(checkpoint_path).endswith(".safetensors"):
        return _load_safetensors(checkpoint_path, device="cpu")
    return _orig_load_state_dict(checkpoint_path, *a, **k)


_oc_factory.load_state_dict = _load_state_dict

config = OmegaConf.to_container(OmegaConf.load("generative_models/configs/unclip6.yaml"), resolve=True)
unclip_params = config["model"]["params"]
network_config = unclip_params["network_config"]
first_stage_config = unclip_params["first_stage_config"]
conditioner_config = unclip_params["conditioner_config"]
sampler_config = unclip_params["sampler_config"]

first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
sampler_config['params']['num_steps'] = 38
# xformers has no CUDA kernels here: use PyTorch attention (same math)
network_config['params']['spatial_transformer_attn_type'] = 'softmax'
first_stage_config['params']['ddconfig']['attn_type'] = 'vanilla'
for emb in conditioner_config['params']['emb_models']:
    if emb['target'].endswith('FrozenOpenCLIPImageEmbedder'):
        emb['params']['version'] = args.bigg_safetensors

diffusion_engine = DiffusionEngine(network_config=network_config,
                                   denoiser_config=unclip_params["denoiser_config"],
                                   first_stage_config=first_stage_config,
                                   conditioner_config=conditioner_config,
                                   sampler_config=sampler_config,
                                   scale_factor=unclip_params["scale_factor"],
                                   disable_first_stage_autocast=unclip_params["disable_first_stage_autocast"])
diffusion_engine.eval().requires_grad_(False)
diffusion_engine.to(device)

unclip_ckpt = args.unclip_ckpt or f'{cache_dir}/unclip6_epoch0_step110000.ckpt'
ckpt = torch.load(unclip_ckpt, map_location='cpu', weights_only=False)
diffusion_engine.load_state_dict(ckpt['state_dict'])
del ckpt

batch = {"jpg": torch.randn(1, 3, 1, 1).to(device),  # placeholder, not used
         "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
         "crop_coords_top_left": torch.zeros(1, 2).to(device)}
vector_suffix = diffusion_engine.conditioner(batch)["vector"].to(device)
print("vector_suffix", vector_suffix.shape)
stage("unCLIP loaded")

# ---- inference ----
# Each shard of --shard_size images is saved on its own so a preempted job can --resume (per-image seeding keeps results identical).
uniq_run = uniq_all[:args.max_images] if args.max_images > 0 else uniq_all
n = len(uniq_run)
shard_dir = f"{out_dir}/partial"
os.makedirs(shard_dir, exist_ok=True)
shard_path = lambda s0: f"{shard_dir}/shard_{s0:05d}.pt"
t_prior = t_cap = t_unclip = 0.0
n_resumed = 0
resize256 = transforms.Resize((256, 256))

for s0 in range(0, n, args.shard_size):
    if args.resume and os.path.exists(shard_path(s0)):
        n_resumed += len(uniq_run[s0:s0 + args.shard_size])
        print(f"shard {s0} already done, skipping", flush=True)
        continue
    sh = {k: [] for k in ["recons", "blurry", "clipvoxels", "prior_out", "betas_avg"]}
    sh["predcaptions"] = []
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for uniq_img in tqdm(uniq_run[s0:s0 + args.shard_size], desc=f"shard {s0}"):
            locs = np.where(test_images_idx == uniq_img)[0]
            if len(locs) == 1:
                locs = locs.repeat(3)
            elif len(locs) == 2:
                locs = locs.repeat(2)[:3]
            assert len(locs) == 3
            voxel_cpu = test_voxels[None, locs]  # 1, 3 reps, num_voxels
            sh["betas_avg"].append(torch.mean(voxel_cpu, dim=1))  # same averaging as final_evaluations.ipynb
            voxel = voxel_cpu.to(device)

            for rep in range(3):
                backbone0, clip_voxels0, blurry_image_enc0 = model.backbone(model.ridge(voxel[:, [rep]], 0))
                if rep == 0:
                    clip_voxels, backbone, blurry_image_enc = clip_voxels0, backbone0, blurry_image_enc0[0]
                else:
                    clip_voxels += clip_voxels0
                    backbone += backbone0
                    blurry_image_enc += blurry_image_enc0[0]
            clip_voxels /= 3
            backbone /= 3
            blurry_image_enc /= 3
            sh["clipvoxels"].append(clip_voxels.cpu())

            t = time.time()
            torch.manual_seed(seed * 1000003 + int(uniq_img))
            prior_out = model.diffusion_prior.p_sample_loop(backbone.shape,
                                                            text_cond=dict(text_embed=backbone),
                                                            cond_scale=1., timesteps=20)
            t_prior += time.time() - t
            sh["prior_out"].append(prior_out.half().cpu())

            t = time.time()
            generated_ids = clip_text_model.generate(pixel_values=clip_convert(prior_out), max_length=20)
            sh["predcaptions"].append(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
            t_cap += time.time() - t

            t = time.time()
            torch.manual_seed(seed * 1000003 + int(uniq_img) + 1)
            samples = utils.unclip_recon(prior_out[[0]], diffusion_engine, vector_suffix, num_samples=1)
            sh["recons"].append(resize256(samples.cpu()).float())  # resized per image; same result as resizing at the end
            t_unclip += time.time() - t

            if args.blurry_recon:
                blurred_image = (autoenc.decode(blurry_image_enc / 0.18215).sample / 2 + 0.5).clamp(0, 1)
                sh["blurry"].append(resize256(blurred_image.cpu()).float())
    torch.save({k: (torch.vstack(v) if k != "predcaptions" else v) for k, v in sh.items() if len(v)}, shard_path(s0) + ".tmp")
    os.replace(shard_path(s0) + ".tmp", shard_path(s0))  # a shard file exists only if complete
    print(f"saved shard {s0}-{s0 + len(sh['predcaptions']) - 1}; first captions: {sh['predcaptions'][:3]}", flush=True)

n_done = n - n_resumed
print(f"per image (this run, {n_done} images): prior {t_prior / max(n_done, 1):.2f}s, caption {t_cap / max(n_done, 1):.2f}s, unCLIP {t_unclip / max(n_done, 1):.2f}s", flush=True)
print("peak GPU mem GB:", torch.cuda.max_memory_allocated() / 1e9)
stage("inference done")

# ---- assemble shards and save (upstream format, plus extras) ----
shards = [torch.load(shard_path(s0), weights_only=False) for s0 in range(0, n, args.shard_size)]
cat = lambda k: torch.vstack([sh[k] for sh in shards])
all_recons = cat("recons")
all_predcaptions = np.array([c for sh in shards for c in sh["predcaptions"]])
assert len(all_recons) == len(all_predcaptions) == n
print(all_recons.shape)
if args.blurry_recon:
    torch.save(cat("blurry"), f"{out_dir}/{out_name}_all_blurryrecons.pt")
torch.save(all_recons, f"{out_dir}/{out_name}_all_recons.pt")
torch.save(all_predcaptions, f"{out_dir}/{out_name}_all_predcaptions.pt")
torch.save(cat("clipvoxels"), f"{out_dir}/{out_name}_all_clipvoxels.pt")
torch.save(cat("prior_out"), f"{out_dir}/{out_name}_all_prior_out.pt")
torch.save(cat("betas_avg"), f"{out_dir}/{out_name}_test_betas_avg.pt")
np.save(f"{out_dir}/{out_name}_image_idx.npy", uniq_run)


# ---- run record ----
def _git(*a):
    try:
        return subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))), *a],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


ckpt_paths = {"mindeye2_last": ckpt_file, "unclip": unclip_ckpt, "bigG_openclip": args.bigg_safetensors,
              "bigG_to_L": f"{cache_dir}/bigG_to_L_epoch8.pth", "betas": f'{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5'}
record = {
    "out_name": out_name, "model_name": args.model_name, "subj": subj, "n_images": int(n), "n_resumed_from_shards": int(n_resumed),
    "seed": seed, "seed_rule": "torch.manual_seed(seed*1000003 + coco_idx) before the diffusion prior; seed*1000003 + coco_idx + 1 before unCLIP",
    "unclip_sampler_steps": 38, "prior_timesteps": 20, "prior_cond_scale": 1.0, "unclip_offset_noise_level": 0.04,
    "blurry_recon": args.blurry_recon, "recon_resolution": 256,
    "precision": "model weights fp32; torch.cuda.amp.autocast(dtype=float16); TF32 matmul enabled",
    "attention": "unCLIP spatial_transformer_attn_type=softmax, first-stage attn_type=vanilla (PyTorch attention)",
    "checkpoints": {k: {"path": v, "bytes": os.path.getsize(v)} for k, v in ckpt_paths.items()},
    "versions": {p: md.version(p) for p in ["torch", "torchvision", "diffusers", "transformers", "open-clip-torch", "numpy", "h5py"]},
    "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
    "date": datetime.datetime.now().isoformat(timespec="seconds"), "gpu": torch.cuda.get_device_name(0),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "timings_s": {**stage_times, "prior_total": round(t_prior, 1), "caption_total": round(t_cap, 1), "unclip_total": round(t_unclip, 1)},
    "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "order_check_passed": order_ok,
}
with open(f"{out_dir}/{out_name}_run_record.json", "w") as fj:
    json.dump(record, fj, indent=1)
for s0 in range(0, n, args.shard_size):
    os.remove(shard_path(s0))
os.rmdir(shard_dir)
print(f"saved {out_name} outputs to {out_dir}")
