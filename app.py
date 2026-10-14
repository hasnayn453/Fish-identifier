#!/usr/bin/env python3
"""
Fish Species Classifier - Smart Gradio Web UI

Three-tier classification:
1. High confidence (>75%) → Identify species
2. Medium confidence (40-75%) → Fish detected, species not in database
3. Low confidence (<40%) → Not a fish
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import gradio as gr


# ============================================================================
# Config
# ============================================================================

DEFAULT_MODEL_DIR = "models"
DEFAULT_PORT = 7860
DEFAULT_HOST = "127.0.0.1"

# Confidence thresholds
HIGH_CONFIDENCE = 75.0    # Identify species
MEDIUM_CONFIDENCE = 40.0  # Fish but unknown species
# Below 40% = Not a fish

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPPORTED_MODELS = ["mobilenet_v3_small", "mobilenet_v3_large", "resnet18"]


# ============================================================================
# Device
# ============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# Model Loading
# ============================================================================

def create_model(model_name: str, num_classes: int) -> nn.Module:
    """Recreate model architecture (no pretrained weights)."""
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(512, num_classes)
        )
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(512, num_classes)
        )
    elif model_name == "resnet18":
        model = models.resnet18(weights=None)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(512, num_classes)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def find_latest_model(model_dir: str = DEFAULT_MODEL_DIR) -> Optional[str]:
    """Find the most recent model checkpoint in models/ folder."""
    path = Path(model_dir)
    if not path.exists():
        return None
    checkpoints = list(path.glob("*_fish_classifier.pth"))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda p: p.stat().st_mtime))


def load_model(model_path: str, device: torch.device) -> Tuple[nn.Module, List[str], int, str]:
    """Load model from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[Load] {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    class_names = ckpt["class_names"]
    model_name = ckpt["model_name"]
    input_size = ckpt["input_size"]
    best_acc = ckpt.get("best_acc", 0.0)

    model = create_model(model_name, len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    print(f"[OK] {model_name} | {len(class_names)} classes | {input_size}px | val_acc={best_acc:.4f}")
    return model, class_names, input_size, model_name


# ============================================================================
# Inference
# ============================================================================

def get_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.15)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def predict(model: nn.Module, img: Image.Image, class_names: List[str],
            transform: transforms.Compose, device: torch.device, input_size: int = 224) -> Tuple[str, float, str]:
    """
    Run inference with three-tier classification.
    
    Returns:
        - result_text: Formatted result message
        - confidence: Confidence percentage
        - status: "identified", "unknown_species", "not_fish"
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    out = model(x)
    probs = F.softmax(out, dim=1)
    top_prob, top_idx = torch.max(probs, dim=1)
    
    class_name = class_names[top_idx.item()]
    confidence = float(top_prob.item()) * 100
    
    # Three-tier classification
    if confidence >= HIGH_CONFIDENCE:
        # High confidence: Species identified
        result_text = f"""## ✅ **Species Identified: {class_name}**

**Confidence:** {confidence:.1f}% 🟢 (Very High)

The fish matches a species in our database.
"""
        status = "identified"
    elif confidence >= MEDIUM_CONFIDENCE:
        # Medium confidence: Fish detected but species not in database
        result_text = f"""## 🐟 **Fish Detected - Species Not in Database**

**Confidence:** {confidence:.1f}% 🟡 (Medium)

The image appears to be a fish, but the species is not among the {len(class_names)} species in our database.

**Top match:** {class_name} ({confidence:.1f}%)
"""
        status = "unknown_species"
    else:
        # Low confidence: Not a fish
        result_text = f"""## ❌ **Not a Fish**

**Confidence:** {confidence:.1f}% 🔴 (Low)

The image does not appear to be a fish from our known species.

**Top match:** {class_name} ({confidence:.1f}%) - below recognition threshold
"""
        status = "not_fish"
    
    result_text += f"\n---\n*Model: {len(class_names)} species • {input_size}px input • {device.type.upper()}*"
    return result_text, confidence, status


# ============================================================================
# Gradio UI
# ============================================================================

def create_ui(model, class_names, input_size, device):
    transform = get_transform(input_size)

    def classify(img):
        if img is None:
            return None, "Upload an image to identify the fish."
        try:
            result_text, confidence, status = predict(model, img, class_names, transform, device, input_size)
            return img, result_text
        except Exception as e:
            return img, f"Error: {e}"

    def clear():
        return None, "Upload an image to identify the fish."

    css = """
    .gradio-container { max-width: 700px !important; margin: 0 auto; }
    #result_box { min-height: 220px; font-size: 16px; }
    """

    with gr.Blocks(css=css, title="Fish Species Classifier") as demo:
        gr.Markdown("# 🐟 Fish Species Classifier\n**Upload a photo - I'll tell you if it's a fish and identify the species**")

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(
                    type="pil", 
                    label="Drag & drop or click to upload", 
                    height=380
                )
                with gr.Row():
                    btn = gr.Button("🔍 Identify", variant="primary", scale=2, size="lg")
                    clr = gr.Button("Clear", variant="secondary", scale=1)
                gr.Markdown("**Formats:** JPG, PNG, BMP, TIFF, WebP")

            with gr.Column(scale=1):
                gr.Markdown("### Result")
                result_box = gr.Markdown(
                    "Upload an image and click **Identify**.", 
                    elem_id="result_box"
                )
                preview = gr.Image(label="Preview", height=280, interactive=False)

        with gr.Accordion("ℹ️ How It Works", open=False):
            gr.Markdown(f"""
            **Three-Tier Classification:**
            
            | Confidence | Result |
            |------------|--------|
            | **≥ {HIGH_CONFIDENCE}%** | ✅ Species identified |
            | **{MEDIUM_CONFIDENCE}% - {HIGH_CONFIDENCE}%** | 🐟 Fish detected, species not in database |
            | **< {MEDIUM_CONFIDENCE}%** | ❌ Not a fish |
            
            **Database:** {len(class_names)} fish species
            
            **Model:** {model.__class__.__name__} • {input_size}px • {device.type.upper()}
            """)

        btn.click(classify, [image_in], [preview, result_box])
        clr.click(clear, [], [image_in, result_box])
        image_in.change(lambda i: classify(i) if i else (None, "Upload an image to identify the fish."),
                       [image_in], [preview, result_box])

    return demo


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fish Species Classifier Web UI")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="Model folder")
    parser.add_argument("--model_path", default=None, help="Specific model file (overrides auto-detect)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--share", action="store_true", help="Public Gradio link")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"[Device] {device}")

    # Find model
    model_path = args.model_path or find_latest_model(args.model_dir)
    if not model_path:
        print(f"[Error] No model found in {args.model_dir}/")
        print("  Train first: python train.py")
        print("  Or specify: python app.py --model_path path/to/model.pth")
        sys.exit(1)

    # Load
    try:
        model, class_names, input_size, model_name = load_model(model_path, device)
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        sys.exit(1)

    # Launch
    print(f"\n{'='*50}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Thresholds: High>={HIGH_CONFIDENCE}% | Medium>={MEDIUM_CONFIDENCE}% | Low<{MEDIUM_CONFIDENCE}%")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*50}\n")

    demo = create_ui(model, class_names, input_size, device)
    try:
        demo.launch(
            server_name=args.host, server_port=args.port,
            share=args.share, debug=args.debug,
            show_error=True, quiet=not args.debug,
            inbrowser=True, theme=gr.themes.Soft()
        )
    except KeyboardInterrupt:
        print("\n[Stop] Server stopped.")
    except OSError as e:
        if "address already in use" in str(e).lower():
            print(f"\n[Error] Port {args.port} in use. Try: python app.py --port {args.port+1}")
        else:
            print(f"\n[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()