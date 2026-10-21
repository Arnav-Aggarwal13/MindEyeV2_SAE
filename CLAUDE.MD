# CLAUDE.md: MindEye2 + SAE on the MIT ORCD Engaging cluster

(Rename to `CLAUDE.md` and put it in the root of a fresh clone of the ORIGINAL repo, https://github.com/MedARC-AI/MindEyeV2, MindEye2, ICML 2024, arXiv:2403.11207. Not a fork.)

Status of this file: written 2026-10-21 from work done on an old local machine (2x TITAN RTX 24 GB, sm_75), where `recon_inference` ran end to end for ONE test image after several fixes (listed at the bottom). Never run on this cluster. `final_evaluations.ipynb` was read, never executed. Numbers marked "estimate" are not measured. I could not see the cluster filesystem when writing this, so every cluster path below is either copied from Arnav's notes for his other projects or marked CONFIRM. Do not invent paths.

## 1. Goal
Interpretability project: build SAE tooling to characterize the information in the representations MindEye2 builds. The trained model is an object of study; reconstruction benchmarks are the measuring stick here, not the objective.

This phase: take the pretrained MindEye2 (subject 1) and measure what happens to its outputs when the diffusion-prior output embedding (`prior_out`) is passed through an SAE before being decoded. Three conditions on the full subject-1 test set (3000 trials = 1000 unique images x 3 repetitions; the loop averages the 3 reps, the script prints the unique count):
1. `base`: pretrained MindEye2, unchanged. All upstream metrics, plus new embedding metrics.
2. `sae_pre`: `prior_out` replaced by `SAE_pre(prior_out)`, where SAE_pre is Arnav's SAE already trained on real-image OpenCLIP bigG tokens. Same decode path, same metrics.
3. `sae_ft`: same, with the SAE finetuned on MindEye-generated `prior_out` from the TRAIN set.
- Optional controls (separate SAE loss from brain noise): `oracle` = true `clip_target` -> unCLIP; `sae_pre_oracle` = `SAE_pre(clip_target)` -> unCLIP. Plus a second `base` run with a different unCLIP seed (noise floor).
- Decisions already made by Arnav:
  - Regular (base unCLIP) recons only. No enhanced/SDXL-img2img recons: it adds generative-prior content (a confound) and needs a community checkpoint (`zavychromaxl_v30.safetensors`) that is not available.
  - Start with the diffusion-prior output `prior_out`. `backbone` is informational only.
  - Image<->brain retrieval must be computed from the diffusion-prior output embeddings too (not only from `clip_voxels`, which upstream uses). See Metrics.
  - SAE/test-image data leakage is ruled out (confirmed). Keep it that way: never finetune the SAE or select anything using test images.
  - Checkpoint: `final_subj01_pretrained_40sess_24bs` (subject 1, pretrained on the other subjects, finetuned on subject 1's 40 sessions): `hidden_dim=4096`, `n_blocks=4`. Verified from the checkpoint tensors: ridge weight `[4096, 15724]`, 4 mixer blocks, `backbone_linear` `[425984, 4096]`.
- Defaults the plan assumes (recommended, not contested; confirm before relying on them): finetune data = TRAIN-set `prior_out` averaged over the 3 reps of each image (~9k images), validated on held-out TRAIN images, never on test; unCLIP noise seeded per image; `prior_out` sampled once and cached.

## 2. What is in the repo (original MedARC-AI/MindEyeV2)
```
README.md, LICENSE, figs/
src/
  Train.ipynb                      training (pretrain + finetune)
  recon_inference.ipynb            voxels -> CLIP embed -> recons + captions
  enhanced_recon_inference.ipynb   SDXL img2img refinement (NOT used here)
  final_evaluations.ipynb          all the metrics (the spec for Stage C below)
  dataset_creation.ipynb           how the dataset was built; reference only, never rerun
  models.py                        BrainNetwork, BrainDiffusionPrior, PriorNetwork, GNet8_Encoder (+ unused Clipper)
  utils.py                         losses, MixCo, metrics, unclip_recon, topk, batchwise_cosine_similarity
  modeling_git.py                  VENDORED HF GIT (patched to accept CLIP embeddings). Do not refactor.
  autoencoder/convnext.py          frozen ConvNeXt-XL teacher for the low-level head
  generative_models/               VENDORED Stability SGM, incl. configs/unclip6.yaml. Do not refactor.
  setup.sh                         pip env script. DO NOT RUN (see section 3).
  accel.slurm, recon_inference.slurm   example job scripts from a different cluster; reference only
```
- Notebooks are the source of truth upstream; `jupyter nbconvert X.ipynb --to python` makes build artifacts. For this work do NOT edit or run the notebooks: port the needed logic into new scripts under `src/interp/` (see Plan) so upstream files stay pristine.
- What the GitHub repo does NOT contain (comes from the HF dataset https://huggingface.co/datasets/pscotti/mindeyev2, ~100 GB for everything): the NSD data, the WebDataset shards, and the shared checkpoints. Details in section 3.
- Never read, list, glob or enumerate the NSD data directory (`$NSD_DATA`, see section 4). It is ~100 GB. Reach it only through the documented `h5py` / `webdataset` paths with explicit filenames. Same for any copy or symlink of it.

## 3. Setting up (fresh clone)
```
git clone https://github.com/MedARC-AI/MindEyeV2.git MindEyeV2 && cd MindEyeV2/src
```
- Git: `origin` is MedARC's repo. Never push there. Ask Arnav which remote to use (his GitHub user is Arnav-Aggarwal13) before any commit or push. New code goes in `src/interp/`. Large files never go in git.
- Env (uv, not `setup.sh`): `setup.sh` pins `torch==2.1.0` and does not match the environment that actually worked. Arnav provides `requirements-working.txt` (239 packages frozen from the working local env; copy it from him, it is not in the repo). Not yet run on the cluster:
  ```
  uv venv fmri --python 3.11
  uv pip install --python fmri/bin/python -r requirements-working.txt
  ```
  - `clip @ git+https://github.com/openai/CLIP.git@d05afc4...` needs GitHub access at install time.
  - Key pins that were load-bearing: `diffusers==0.23.0`, `transformers==4.37.2`, `huggingface-hub==0.25.2` (diffusers 0.23.0 imports `cached_download`, removed in hub 0.26), `setuptools==80.10.2` (must be < 81: later versions have no `pkg_resources`, which pytorch_lightning needs), `open-clip-torch==2.24.0`, `accelerate==0.24.1`, `pytorch-lightning==2.0.1`, `kornia==0.7.1`, `scikit-image==0.22.0`, `numpy==1.26.4`, `h5py==3.10.0`, `webdataset==0.2.73`, `dalle2-pytorch==1.15.6`, `deepspeed==0.13.1`, `omegaconf==2.3.0`, `sentence-transformers==2.5.1`, `evaluate==0.4.1`, `nltk==3.8.1`, `torchmetrics==1.3.0.post0`, `timm==1.0.29`.
  - torch was `2.14.0+cu130` / torchvision `0.29.0` (the freeze drops the CUDA tag). If the cluster driver cannot run cu130, pick a torch wheel that matches the driver, then re-run a smoke test.
  - `xformers==0.0.22.post7` is installed but UNUSABLE (built for torch 2.1; no CUDA kernels). Never rely on it; use the attention override in section 8.
  - Activate with `source fmri/bin/activate`, run everything from `src/` (all paths resolve relative to `src/`).
- Data and shared checkpoints: the README download script puts EVERYTHING flat in one directory (`hf_hub_download(..., local_dir=...)`, excluding `train_logs`/`evals` except three files). Keep that flat layout in one directory `$NSD_DATA` and pass it as BOTH `--data_path` and `--cache_dir` (upstream defaults both to the cwd). Needed files (explicit names only):
  - `betas_all_subj01_fp32_renorm.hdf5` (~1.9 GB, loaded fully into RAM)
  - `wds/subj01/new_test/0.tar` and `wds/subj01/train/` shards (behavioral arrays only; voxels and images are indexed from the hdf5 files)
  - `sd_image_var_autoenc.pth`, `bigG_to_L_epoch8.pth`, `unclip6_epoch0_step110000.ckpt`
  - `gnet_multisubject.pt`, `brain_region_masks.hdf5` (eval)
  - `evals/all_images.pt`, `evals/all_captions.pt`, `evals/all_git_generated_captions.pt` (eval ground truth; images and captions for the 1000 test images)
  - `coco_images_224_float16.hdf5` (22 GB; upstream opens it lazily but our scripts do not need it; skip it)
- Pretrained model: `train_logs/final_subj01_pretrained_40sess_24bs/last.pth` from the HF repo's `train_logs/` folder (8.3 GB, measured). The scripts read it from `../train_logs/<model_name>/last.pth` relative to `src/`. `train_logs/` can be a symlink to a directory on home or pool. The SAE is separate (section 4).
- Outputs follow upstream: `src/evals/<model_name>/<model_name>_all_{recons,blurryrecons,predcaptions,clipvoxels}.pt`, `src/tables/<name>.csv`.
- Offline compute nodes: prefetch on a node with internet (ask Arnav before large downloads, see section 5): open_clip bigG `laion2b_s39b_b160k` (~10 GB, loaded twice: standalone embedder and inside the unCLIP conditioner), `microsoft/git-large-coco`, HF `openai/clip-vit-base-patch32` and `-large-patch14`, `sentence-transformers/all-MiniLM-L6-v2`, `clip.load("ViT-L/14")` (OpenAI CDN), `torch.hub` `facebookresearch/swav:main` resnet50, torchvision weights (alexnet, inception_v3, efficientnet_b1), `evaluate` meteor/rouge plus NLTK wordnet data. Point `HF_HOME` and `TORCH_HOME` at a persistent directory.

## 4. Where things live on the cluster (MIT ORCD Engaging)
What Arnav's notes for his other projects (SAETraining, Prisma-SAE) establish:
- Home: 200 GB quota, the only backed-up location. Repos, each repo's `.venv` (several GB each), run outputs (`$RUNS_DIR`, e.g. `~/prisma_runs`), and the SAE checkpoints live here. Check usage with `~/orcd/.quota`; the quota is shared across all his projects.
- Pool: slower HDD-backed storage for big read-mostly datasets (SAETraining's Open Images thumbnails live here). Random small reads from many files can starve a GPU; copy to scratch if that happens.
- Scratch: capped working storage for big regenerable intermediates (SAETraining's activation shards live here). Treat pool and scratch as not backed up.
- HF cache: SAETraining's `HF_HOME` points at a shared cache so the bigG weights are not downloaded twice. Reuse that cache (read SAETraining's `env.sh` for the value; read-only). The bigG safetensors can load offline via a `MODEL_LOCAL` path there.
- Arnav's SAE (read-only input; separate repo `Prisma-SAE`, which has its own CLAUDE.md): trained with ViT-Prisma on the bigG `ln_post` patch tokens of Open Images. Best checkpoint so far: `$RUNS_DIR/full_tied/5dc3a416-bigg-prisma-sae/n_images_496128.pt` (TopK k=128, expansion 16x, d_sae 26624, raw un-normalized input, val nmse 0.219, ~0% dead latents). CONFIRM this is the intended SAE and get the real `$RUNS_DIR`.
- Never write, move or delete anything in SAETraining or Prisma-SAE, never run their scripts, never source their `env.sh` (read it, copy values).

Placement for THIS project (proposal; CONFIRM with Arnav before downloading anything):
| What | Size (estimate) | Suggested place |
|---|---|---|
| this repo, `fmri` venv, SAE ckpts, `results.jsonl`, metrics, per-image arrays | ~10-15 GB | home |
| `last.pth`, HF/torch caches (bigG, GIT, CLIP, eval nets) | ~8 GB + ~15-20 GB | home, or pool if home is tight (read once into RAM) |
| `$NSD_DATA` (betas, wds shards, unCLIP ckpt, VAE ckpt, GNet, evals ground truth) | ~35-50 GB without the COCO hdf5 | pool (read once at start), or scratch |
| cached embeddings: test (backbone, prior_out, clip_target, ~2.6 GB each fp16) and train (rep-averaged, ~7.7 GB per tensor fp16) | ~15-25 GB | scratch (regenerable) |
| recons per condition (256 px fp32 `.pt`, ~2-5 GB each) | ~12-30 GB total | home if it fits, else scratch |
Total about 90-150 GB. The swing items: single-trial train embeddings (+30 GB), COCO hdf5 (+22 GB), saving 768 px recons (+3.5 GB per condition).

## 5. Using this cluster (rules)
- Do NOT work on the login node: no model loading, no inference, no heavy Python, no big data reads. Use it for editing, git, and light shell only.
- Test things with a CPU job first (`srun`/`sbatch` with the CPU account/qos below): import checks, `py_compile`, metric functions on synthetic data, SAE-module tests on random tensors, reading hdf5/webdataset headers, path checks. Small CPU smoke tests are fine to run yourself.
- GPU jobs: only submit after Arnav gives permission for that specific job. Before asking, state the script, the resource request (GPUs, VRAM, RAM, walltime) and the expected runtime. Never launch long jobs, `accelerate launch`, or notebooks on your own.
- Downloads: if compute nodes have no internet (unknown, CONFIRM), prefetching must happen from a node that has it. Ask Arnav before starting any large download. Never download into the repo dir or git.
- Every Slurm job needs: `#SBATCH --account=mit_amf_standard_cpu` and `#SBATCH --qos=mit_amf_standard_cpu` for CPU jobs; `#SBATCH --account=mit_amf_standard_gpu` and `#SBATCH --qos=mit_amf_standard_gpu` for GPU jobs. Copy partition/resource lines from SAETraining's `slurm/train.sbatch` (read-only), not from the upstream `*.slurm` files (another cluster). `logs/` must exist before `sbatch`. Use an `env.sh` of this project's own paths (copy values, do not source another repo's). Overrides via `--export=ALL,VAR=...`.
- Resource guess for the GPU stages (estimate): one GPU with >= 48 GB VRAM (80 GB preferred), ~64 GB job RAM. Running everything on one GPU at fp32 weights needs roughly 40 GB: MindEye2 ~8 GB, unCLIP stack ~19 GB, bigG embedder ~7.6 GB, GIT ~1.6 GB, plus activations. Images are independent, so a job array over image shards is the easy speedup.
- W&B (if used): `WANDB_MODE=offline` when nodes lack internet.

## 6. My personalization
- Any code you write, summarize in the chat what it does and why
- Any code that runs for a non trivial amount of time, add progress bars or print statements in the log to track progress

## 7. Plan (ordered; each step lists its gate)
Code: new scripts in `src/interp/`, one per stage. Seeds: global `--seed 42`. `prior_out` is sampled ONCE and cached, so prior noise is identical across conditions. unCLIP noise is seeded per image (`torch.manual_seed(f(seed, coco_idx))` right before each `unclip_recon`), because upstream's `p_sample` ignores its `generator` argument and `unclip_recon` draws unseeded `torch.randn`. Precision and attention settings are identical in every condition. Condition names must NOT contain "enhanced" (the upstream eval notebook blends blurry recons into any name that does).

Stages (decoupled by cached tensors):
- A, collect (GPU, MindEye2 only, no unCLIP): ridge + backbone (3 reps averaged) + prior sampling (`p_sample_loop`, 20 DDIM steps). Save in `np.unique(test_images_idx)` order plus the index array: `backbone`, `prior_out`, `clip_voxels`, blurry latents (fp16 `(1000,256,1664)`), and `clip_target` computed from `evals/all_images.pt`. Then the same for the TRAIN set (3-rep averaged, `backbone` and `prior_out` only).
- B, decode (GPU): input = cached embedding file + condition name + optional SAE. Runs `clip_convert` + GIT for captions and `unclip_recon` for images, one image at a time, per-image seeded. Saves the upstream-format `evals/<cond>/...` files. The expensive stage (unCLIP).
- C, eval (GPU): port of `final_evaluations.ipynb` plus the new embedding metrics. Saves a metrics table, per-image arrays (`.npz`) and a row in `results.jsonl`.
- `compare.py`: paired per-image differences with bootstrap CIs over the 1000 images.

Steps:
0. Get from Arnav: `requirements-working.txt`, the pretrained `last.pth`, and decide data/model placement (section 4).
1. Cluster setup: clone, env, prefetch, `env.sh`, sbatch templates. CPU-job smoke tests. Then a 20-image GPU smoke test through A-C (with permission). Time it.
2. SAE access (Prisma-SAE is read-only): do NOT install vit-prisma or import `prisma_bigg` in this env (its dependencies conflict with these pins). Export the SAE once, in the Prisma-SAE env, to plain tensors (`W_enc`, `b_enc` if present, `W_dec`, `b_dec`, `k`, config). Here, write a standalone SAE module and load the tensors. Prisma's own loading needs `import prisma_bigg` first (the checkpoint pickles `HookedSAEConfig`) and `SparseAutoencoder.load_from_pretrained(path, current_cfg={"_device": ...})`. Match Prisma's `SparseAutoencoder` forward at its pinned commit; do not guess the formula.
3. Stage A on test and train. Run Gates 3-5.
4. Stage C port + embedding metrics. Run first on cached tensors for `base` (embedding metrics and captions need no unCLIP).
5. `base` through B and C (Gates 1-2). Repeat with a second unCLIP seed.
6. `sae_pre` through B and C.
7. SAE finetune on train `prior_out`. Preferred: a new `embeddings` data route in Prisma-SAE (reads cached `(N,256,1664)` fp16 tensors, serves shuffled token batches into Prisma's trainer), so training code and checkpoint format stay in one place; that is a task in the Prisma-SAE repo, ask Arnav. Alternative: standalone loop here replicating Prisma's step (renormalize decoder rows to unit norm, grad clip 1.0, remove the gradient component parallel to decoder rows, Adam, loss = squared error divided by each token's L2 norm about the batch mean). Start finetune lr at 1e-4 (from-scratch used 1e-3), raw input (no normalization). Validate on held-out TRAIN images. Re-run Gates 3-4 on the finetuned SAE.
8. `sae_ft` through B and C.
9. Optional controls, then `compare.py` and a write-up.

Gates (must pass before spending GPU hours on the next step):
1. Baseline fidelity: `base` metrics land close to the published subject-1 numbers (paper/README), within a few points. Expect small differences (new_test set, TF32, SDP attention). If far off, stop and debug before any SAE condition.
2. Determinism: decoding the same image twice gives identical recons and captions.
3. SAE parity: the standalone module vs Prisma's `SparseAutoencoder` on >= 1000 real tokens, fp32, max abs diff < 1e-4, and per-image output independent of batch composition.
4. SAE reconstruction on the test set: nmse (SSE(x - x_hat) / SSE(x - mean)) and cosine for `x` in {`clip_target`, `prior_out`, `backbone`}, plus a scale check (per-token norm, mean, std of `prior_out` vs `clip_target`). `clip_target` should be near the Open Images val value (~0.22; COCO may differ). If `prior_out` is far worse or off-scale, report it and discuss before Condition 2. Never silently rescale.
5. Ordering: `all_images.pt` rows, cached embeddings and recons must share the same image order (assert the index array; sanity-check that cosine between `clip_img_embedder(all_images)` and `prior_out` is diagonal-dominant).

Rough cost (estimates, A100-80GB class; H100 ~1.5x faster, the old 24 GB Titan RTX ~4-5x slower): one 1000-image generation pass ~2 h unCLIP (1.5-3 h) + captions ~5 min + eval ~0.5 h. Core runs (3 conditions + collection + finetune) ~9-10 GPU-h; with controls and a baseline repeat ~16-17 GPU-h; with debugging budget ~20-25 GPU-h. Collection: test ~15 min, train ~0.5-1 h. SAE finetune: well under 1 h. A 4-GPU job array cuts a pass to ~30-40 min. Calendar ~1.5-2 weeks, mostly coding, debugging and queue time.

## 8. What was learned about the repo (and what is broken)
The model:
```
fMRI betas (B, 1, n_vox)                      n_vox = 15724 for subj01, read at runtime, never hardcode
  -> [TRAINED] RidgeRegression, one nn.Linear per subject      -> (B, 1, 4096)
  -> [TRAINED] BrainNetwork (MLP-Mixer + heads)
       backbone (B,256,1664)   clip_voxels (B,256,1664) = clip_proj(backbone)   blurry ((B,4,28,28),(B,49,512))
  -> [TRAINED] BrainDiffusionPrior, conditioned on backbone, denoises toward clip_target   -> prior_out (B,256,1664)
       -> [FROZEN] SDXL unCLIP -> 768x768 reconstruction
       -> [FROZEN] CLIPConverter (Linear 256->257 on the sequence axis, 1664->1024 on the embedding axis) -> GIT-large-coco -> caption
```
Training loss: `prior_scale*MSE(prior)` + `clip_scale*(mixco_nce | soft_clip_loss)` on `clip_voxels` + `blur_scale*(L1 + 0.1*soft_cont)`. Training logs `recon_cossim` / `recon_mse` of `prior_out` vs `clip_target`. Inference per unique test image: average 3 voxel reps through ridge+backbone, `model.diffusion_prior.p_sample_loop(backbone.shape, text_cond=dict(text_embed=backbone), cond_scale=1., timesteps=20)`, then `clip_convert` + `GitForCausalLMClipEmb.generate(pixel_values=..., max_length=20)` and `utils.unclip_recon(...)` (38 sampler steps); outputs saved at 256 px.

Which embedding is which (critical):
- `clip_target = clip_img_embedder(image)`: OpenCLIP ViT-bigG/14 (`laion2b_s39b_b160k`, `output_tokens=True, only_tokens=True`). It is the output after ALL 48 transformer blocks and the final `ln_post`, patch tokens only (CLS dropped), `(B,256,1664)`. The final `proj` matrix is applied only to the pooled CLS vector, never to these tokens, so this is the ViT's last hidden state (width 1664), not the 1280-d joint CLIP space used for text-image retrieval. Verified by tracing `FrozenOpenCLIPImageEmbedder` and open_clip's `VisionTransformer.forward`.
- This is exactly the tensor Arnav's SAE was trained on (Prisma target `hook_ln_final[:, 1:]`, raw, `normalize_activations="none"`). So `clip_target` is in-distribution for the SAE by construction.
- `prior_out`: the diffusion prior's denoised sample, directly MSE/cosine-supervised against `clip_target`. It is what unCLIP and the captioner consume. THIS is the tensor under study and where the SAE is inserted.
- `backbone`: pre-prior brain embedding, same shape, but no direct reconstruction loss against `clip_target`; it only conditions the prior and feeds `clip_proj`.
- `clip_voxels`: contrastively trained head (cosine/ranking, not exact reconstruction). Upstream retrieval uses this one.
- Applying the SAE: `(B,256,1664) -> (B*256,1664) -> SAE -> (B,256,1664)`, per-token TopK, fp32 (activations are fp16 under autocast; cast first), no normalization or rescaling.

Metrics (all per image over the 1000 test images; higher is better unless noted; cell numbers are from `final_evaluations.ipynb`). Keep upstream's protocol so numbers are comparable with the paper.
- Embedding level, NEW (cached tensors, no diffusion): image<->brain retrieval from the diffusion-prior output, same protocol as upstream cell 11 (flatten `256x1664`, L2-normalize, 30 random draws of 300 images, top-1, mean and 95% CI), brain-side = `prior_out` (and `sae_pre(prior_out)`, `sae_ft(prior_out)`), image-side = `clip_target`; both directions with explicit names `brain_to_image` and `image_to_brain`; chance 1/300. Also report it for `backbone` (informational) and for `clip_voxels` (upstream's number, to check our function reproduces upstream). Also per-image cosine (flattened) and per-token cosine vs `clip_target`, MSE, explained variance for each embedding (comparable to the training logs), and SAE diagnostics on `prior_out` vs on `clip_target` (nmse, cosine, fraction of latents active on only one of the two).
- Naming trap: upstream's `fwd_sim = batchwise_cosine_similarity(emb_, emb)` ends in `.T`, so rows are images and its "fwd"/"bwd" labels look swapped relative to their comments. Do not copy the labels; verify on `clip_voxels`.
- Low-level image (recon vs its own original): PixCorr (16; both resized to 425x425, flattened RGB, Pearson r per pair: raw pixel agreement), SSIM (18; grayscale 425x425, Gaussian-window sigma 1.5: local luminance/contrast/structure), AlexNet(2) and AlexNet(5) (20; two-way identification on activations after conv2+ReLU `features.4` and conv5+ReLU `features.11`, flattened spatial maps: early and mid-level features with position retained).
- Two-way identification (shared logic): for each recon, Pearson correlation of its features with its true original and with every other original; score = fraction of the 999 distractors that correlate less than the true one, averaged. Chance 50%.
- High-level image: InceptionV3 (22; two-way identification on 2048-d pooled features: object/scene content, position-invariant), CLIP (24; two-way identification on OpenAI ViT-L/14 `encode_image`, a CLIP independent of the LAION bigG MindEye targets, so not circular), EffNet-B (26) and SwAV (28) (1 - Pearson correlation between pooled deep features of recon and original, averaged; LOWER is better).
- Brain space (30-34): GNet (pretrained image->fMRI encoding model) predicts voxel responses for each recon; per image, Pearson r across the voxels of a region between measured betas (3-rep average, original image) and GNet's prediction for the recon, averaged over images. Regions: `nsd_general`, V1, V2, V3, V4, `higher_vis`. Capped by GNet's own accuracy.
- Captions (37-49): three sets (human COCO captions, GIT on the real image, brain-derived `prior_out` -> CLIPConverter -> GIT). For METEOR, ROUGE-1, ROUGE-L, MiniLM sentence similarity, CLIP-B/32 and CLIP-L/14 text cosine: GROUND (GIT-on-image vs human, the ceiling), ABSOLUTE (brain vs human, brain vs GIT-on-image), RELATIVE (brain-vs-GIT-image divided by the ceiling). A cheap semantic readout of `prior_out` (no unCLIP), so run it first.
- Under an SAE swap everything computed from `prior_out` moves (image metrics, brain correlation, captions, the new embedding metrics). Upstream retrieval (`clip_voxels`) does NOT move, because the SAE never touches that head. Hence the new `prior_out` retrieval.
- Statistics: upstream's retrieval CI uses 30 overlapping subsets and is optimistic. For condition comparisons use paired per-image differences with bootstrap CIs. Run `base` twice with different unCLIP seeds to judge SAE effects against sampling noise.

Non-obvious upstream behavior:
- `text_embed` is the brain embedding: `PriorNetwork.forward` does `brain_embed = text_embed`. No text anywhere in that path.
- Disabled heads return dummy tensors (`c = Tensor([0.])`), not `None`.
- `p_sample` ignores its `generator` for the noise draw; `unclip_recon` uses unseeded `torch.randn` (hence per-image seeding above).
- `ConvnextXL.init_weights` loads with `strict=False` inside a bare try/except: weight-loading failures are silent.
- argparse values become bare module-level globals, and `utils.is_interactive()` switches between a hardcoded `jupyter_args` string and real argparse. Edit both branches or they disagree.
- Inference argparse defaults differ from training; `hidden_dim` defaults to 2048 but this checkpoint needs 4096. Mismatched `hidden_dim`/`n_blocks`/`blurry_recon` gives shape errors or silent wrong behavior under `strict=False` (the DeepSpeed-format fallback loads with `strict=False`).
- Training silently drops batches with duplicate images (`if len(image0) != len(image_idx): continue`). Irrelevant here, relevant if retraining.

Things that WILL break on a fresh upstream clone (in the order they were hit), and the fix:
1. `ModuleNotFoundError: pkg_resources` importing `sgm`/`pytorch_lightning`: `setuptools` too new. Pin `setuptools<81`.
2. `ImportError: cannot import name 'cached_download'` from `huggingface_hub` (via diffusers): pin `huggingface-hub==0.25.2`.
3. `NotImplementedError: No operator found for memory_efficient_attention_forward` (xformers has no CUDA kernels): after loading `unclip6.yaml` in the script, override `network_config['params']['spatial_transformer_attn_type'] = 'softmax'` and `first_stage_config['params']['ddconfig']['attn_type'] = 'vanilla'` (PyTorch SDP attention; same math). Do it next to upstream's own `first_stage_config['target'] = ...` override. Never edit the vendored YAML.
4. Running `recon_inference` as a script (not Jupyter): `plotting` is only assigned when interactive, so `if plotting:` raises `NameError`. Set `plotting = utils.is_interactive()`. Also under Jupyter it deliberately executes a bare `err` after one image (intentional upstream).
5. `--hidden_dim` default 2048 vs checkpoint 4096 (above). Pass `--hidden_dim=4096 --n_blocks=4`.
6. Hardcoded `/weka/proj-medarc/...` paths in the notebooks' `jupyter_args`, and in `final_evaluations`; the eval notebook also loads `evals/...` and `brain_region_masks.hdf5` relative to the cwd. Parameterize.
7. `final_evaluations` SSIM cell: `structural_similarity(..., multichannel=True)` raises TypeError on scikit-image 0.22 (the kwarg was removed). Use `channel_axis` (the inputs are already grayscale).
8. Eval cell outputs: any output name containing "enhanced" triggers `all_recons*.75 + all_blurryrecons*.25`.
9. First-use downloads (internet): `torch.hub` SwAV, `clip.load`, `evaluate.load`, sentence-transformers, HF CLIP. Prefetch.
10. Only needed on 24 GB GPUs (NOT on this cluster; keep the faithful fp32-weights + fp16-autocast path on GPU): the unused standalone `clip_img_embedder` was left off the GPU, GIT and `clip_convert` were moved to CPU (needs `prior_out.float().cpu()` before `clip_convert`), and the unCLIP model and blurry VAE were stored in fp16. If anyone does cast: unclip6.yaml sets `disable_first_stage_autocast: True`, so the unCLIP VAE (`diffusion_engine.first_stage_model`) must stay fp32, `decode_first_stage` needs fp32 input (`samples_z.float()` in `utils.unclip_recon`), and the placeholder `batch` tensors for the conditioner call must match the model dtype. Do not mix settings across conditions.
- Note for the new work: the standalone `clip_img_embedder` is unused in `recon_inference`, but IS needed here to compute `clip_target` from `all_images.pt`.
