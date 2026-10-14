#!/usr/bin/env python3
"""
Download Model from Hugging Face Hub

Downloads the trained fish classifier model from Hugging Face Hub.
Run this after cloning the repo if you don't want to train locally.
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import hf_hub_download


DEFAULT_REPO = "hasnayn453/fish-classifier"
DEFAULT_FILENAME = "fish_classifier.pth"
DEFAULT_OUTPUT = "models/fish_classifier.pth"


def main():
    parser = argparse.ArgumentParser(description="Download fish classifier model from HF Hub")
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO, help="HF Hub repo ID")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="Model filename in repo")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Local output path")
    parser.add_argument("--token", type=str, default=None, help="HF token (for private repos)")

    args = parser.parse_args()

    print(f"Downloading from {args.repo}/{args.filename}...")
    print(f"Output: {args.output}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    try:
        path = hf_hub_download(
            repo_id=args.repo,
            filename=args.filename,
            local_dir=".",
            local_dir_use_symlinks=False,
            token=args.token,
        )

        if str(path) != args.output:
            import shutil
            shutil.move(path, args.output)

        print(f"[OK] Model downloaded to {args.output}")

        # Verify
        import torch
        checkpoint = torch.load(args.output, map_location="cpu", weights_only=False)
        print(f"  Classes: {len(checkpoint.get('class_names', []))}")
        print(f"  Model: {checkpoint.get('model_name', 'unknown')}")
        print(f"  Best Acc: {checkpoint.get('best_acc', 'unknown'):.4f}")

    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("\nTroubleshooting:")
        print(f"  1. Check repo exists: https://huggingface.co/{args.repo}")
        print(f"  2. For private repos, add --token YOUR_HF_TOKEN")
        print(f"  3. Or train locally: python train.py --data_dir dataset")
        sys.exit(1)


if __name__ == "__main__":
    main()