"""Download the frozen networks the image/brain evals use into TORCH_HOME (run as a CPU job; compute nodes have internet)."""
import os
import sys
import traceback
import torch

torch_home = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
print("TORCH_HOME:", torch_home, flush=True)

ok = {}


def step(name, fn):
    try:
        fn()
        ok[name] = True
        print(f"OK   {name}", flush=True)
    except Exception:
        ok[name] = False
        print(f"FAIL {name}", flush=True)
        traceback.print_exc()


from torchvision.models import alexnet, AlexNet_Weights, inception_v3, Inception_V3_Weights, efficientnet_b1, EfficientNet_B1_Weights
step("alexnet", lambda: alexnet(weights=AlexNet_Weights.IMAGENET1K_V1))
step("inception_v3", lambda: inception_v3(weights=Inception_V3_Weights.DEFAULT))
step("efficientnet_b1", lambda: efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT))
step("swav resnet50 (torch.hub)", lambda: torch.hub.load('facebookresearch/swav:main', 'resnet50'))
import clip
step("clip ViT-L/14", lambda: clip.load("ViT-L/14", device="cpu", download_root=f"{torch_home}/clip"))

# ---- caption-eval models / data (HF_HOME, nltk_data) ----
def _nltk():
    import nltk
    for pkg in ["wordnet", "punkt", "omw-1.4"]:
        assert nltk.download(pkg, quiet=True), pkg


def _evaluate():
    import evaluate
    evaluate.load('meteor')
    evaluate.load('rouge')


def _hf_clip(name):
    from transformers import CLIPModel, AutoTokenizer
    CLIPModel.from_pretrained(name)
    AutoTokenizer.from_pretrained(name)


step("nltk data", _nltk)
step("evaluate meteor+rouge", _evaluate)
step("MiniLM sentence model", lambda: __import__("sentence_transformers").SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'))
step("HF CLIP B/32", lambda: _hf_clip("openai/clip-vit-base-patch32"))
step("HF CLIP L/14", lambda: _hf_clip("openai/clip-vit-large-patch14"))

print(ok)
sys.exit(0 if all(ok.values()) else 1)
