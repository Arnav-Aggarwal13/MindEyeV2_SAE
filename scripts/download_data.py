import os, sys
from huggingface_hub import list_repo_files, hf_hub_download

# README download code verbatim, plus the pretrained checkpoint (README excludes train_logs)
repo_id, branch, exclude_dirs, exclude_files = "pscotti/mindeyev2", "main", ["train_logs", "evals"], ["human_trials_mindeye2.ipynb", "subj01_annots.npy", "shared1000.npy"]
include_specific_files = ["evals/all_images.pt", "evals/all_captions.pt", "evals/all_git_generated_captions.pt",
                          "train_logs/final_subj01_pretrained_40sess_24bs/last.pth"]

files = list_repo_files(repo_id, repo_type="dataset", revision=branch)
todo = [f for f in files if (not any(ex in f for ex in exclude_dirs) or f in include_specific_files) and not any(ex in f for ex in exclude_files)]
print(f"{len(todo)} files to fetch into {os.getcwd()}", flush=True)
for i, f in enumerate(todo, 1):
    print(f"[{i}/{len(todo)}] {f}", flush=True)
    hf_hub_download(repo_id, filename=f, repo_type="dataset", revision=branch, local_dir=os.getcwd())
print("DONE", flush=True)
