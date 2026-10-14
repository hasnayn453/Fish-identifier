# Fish Species Classifier 🐟

A complete, local-first pipeline for training and deploying a fish species image classification model using PyTorch and Gradio.

## Features

- **Transfer Learning**: MobileNetV3-Small (default, ~10MB), MobileNetV3-Large, or ResNet18
- **Auto Hardware Detection**: CUDA (NVIDIA), MPS (Apple Silicon), or CPU
- **Modern Web UI**: Gradio interface with drag-and-drop, confidence bars, top-K predictions
- **GitHub Ready**: Repo < 100MB (large files gitignored, model hosted on Hugging Face Hub)
- **Offline-First**: Runs entirely locally after setup — no cloud, no API keys required

---

## Quick Start (After Cloning)

```bash
# 1. Clone the repo
git clone https://github.com/hasnayn453/Fish-identifier.git
cd Fish-identifier

# 2. Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download pre-trained model (or train your own - see below)
python download_model.py

# 5. Launch web UI
python app.py
# Opens http://127.0.0.1:7860
```

---

## Option A: Use Pre-trained Model (Recommended)

Download the model trained on 31 fish species from Hugging Face Hub:

```bash
python download_model.py
# Saves to models/fish_classifier.pth (~10 MB)
```

Then launch the UI:
```bash
python app.py
```

---

## Option B: Train Your Own Model

### 1. Prepare Dataset

Create `dataset/` folder with class subfolders (folder name = class label):

```
dataset/
├── Bangus/
│   ├── img_001.jpg
│   └── ...
├── Tilapia/
│   ├── img_001.jpg
│   └── ...
└── Catfish/
    ├── img_001.png
    └── ...
```

**Requirements:**
- Minimum 30 images per class (100+ recommended)
- Supported: JPG, JPEG, PNG, BMP, TIFF, WebP

### 2. Train

```bash
# GPU (NVIDIA RTX 4060, etc.) - fast
python train.py --data_dir dataset --model_name mobilenet_v3_small --batch_size 64 --epochs 15

# CPU fallback
python train.py --data_dir dataset --model_name mobilenet_v3_small --batch_size 16 --num_workers 0
```

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `dataset` | Dataset root folder |
| `--model_name` | `mobilenet_v3_small` | Architecture: `mobilenet_v3_small`, `mobilenet_v3_large`, `resnet18` |
| `--batch_size` | `32` | Batch size (64 for 8GB VRAM) |
| `--epochs` | `15` | Max epochs |
| `--lr` | `0.001` | Learning rate |
| `--model_dir` | `models` | Output directory (gitignored) |

Model saves to `models/mobilenet_v3_small_fish_classifier.pth` (~10 MB, **gitignored**).

### 3. Run UI

```bash
python app.py
# Auto-detects latest model in models/
```

---

## Windows Batch Files (Double-click to run)

| File | Purpose |
|------|---------|
| `run.bat` | **Interactive menu** - choose train, run UI, download, install |
| `train_gpu.bat` | Direct GPU training (MobileNetV3-Small, batch=64) |
| `run_app.bat` | Launch Gradio web UI |

---

## Project Structure

```
Fish-identifier/
├── .gitignore              # Excludes: venv/, models/, dataset/, *.pth, __pycache__/
├── config.yaml             # Training/inference configuration
├── requirements.txt        # Minimal dependencies (6 packages)
├── download_model.py       # Download model from Hugging Face Hub
├── train.py                # Training script (saves to models/)
├── app.py                  # Gradio web UI (loads from models/)
├── run.bat                 # Windows interactive menu
├── train_gpu.bat           # Windows GPU training
├── run_app.bat             # Windows web UI launcher
├── README.md
└── dataset/                # YOUR DATA HERE (gitignored)
    ├── class_a/
    └── class_b/
```

---

## Model Architectures

| Model | Parameters | Weights Size | Speed | Best For |
|-------|------------|--------------|-------|----------|
| **MobileNetV3-Small** | 2.5M | **~10 MB** | Very Fast | **Default** - mobile/edge, real-time |
| MobileNetV3-Large | 5.5M | ~22 MB | Fast | Balanced |
| ResNet18 | 11.7M | ~45 MB | Fast | Highest accuracy |

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
gradio>=4.0.0
Pillow>=9.0.0
numpy>=1.24.0
tqdm>=4.65.0
huggingface_hub>=0.20.0
```

Install: `pip install -r requirements.txt`

---

## Hugging Face Hub Model Hosting

The pre-trained model is hosted at: **https://huggingface.co/hasnayn453/fish-classifier**

To host your own model:
```bash
# 1. Train your model
python train.py --data_dir dataset

# 2. Install HF CLI
pip install huggingface_hub

# 3. Login
huggingface-cli login

# 4. Upload
huggingface-cli upload your-username/your-repo models/mobilenet_v3_small_fish_classifier.pth
```

Then update `download_model.py` DEFAULT_REPO or use `--repo your-username/your-repo`.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Activate venv: `venv\Scripts\activate` |
| `CUDA out of memory` | Reduce batch: `--batch_size 16` |
| `Model not found` | Run `python download_model.py` or train |
| `Port in use` | Use `--port 8080` |
| `Dataset not found` | Create `dataset/class_name/` folders |
| `nvidia-smi` fails | Run as Administrator or check GPU drivers |

---

## GitHub Clone Verification

After cloning, repo size should be **< 1 MB** (code only):

```bash
git clone https://github.com/hasnayn453/Fish-identifier.git
cd Fish-identifier
du -sh .  # Should show ~500 KB (excluding .git)
```

Large files are **gitignored**:
- `venv/` (virtual environment)
- `models/` (model checkpoints ~10-45 MB)
- `dataset/` (your images ~GB)
- `*.pth`, `*.log`, `__pycache__/`

Users download model via `python download_model.py` or train their own.

---

## License

MIT License — Free for personal and commercial use.

---

## Credits

- **PyTorch** & **torchvision** - Training framework
- **Gradio** - Web interface
- **Hugging Face Hub** - Model hosting
- ImageNet pre-trained weights

---

**Happy classifying! 🎣**