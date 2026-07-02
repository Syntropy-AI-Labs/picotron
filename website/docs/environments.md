# Environments & Platform Setup

Welcome to the Picotron setup guide. This document provides step-by-step instructions for running Picotron in local environments and high-performance cloud platforms like Kaggle and Google Colab.

---

## ☁️ Cloud Platforms (Kaggle & Google Colab)

Cloud instances offer access to powerful accelerators. To prevent common disk cache, package lookup, and execution speed issues, follow this setup process.

### Kaggle 2xT4 Accelerator Setup
Kaggle provides dual NVIDIA T4 GPUs supporting Distributed Data Parallel (DDP) execution. To configure the workspace:

```bash
# Set working workspace and clone repository
%cd /kaggle/working
!rm -rf picotron
!git clone https://github.com/Syntropy-AI-Labs/picotron.git

# Install in editable mode
%cd picotron
!pip install -e .
```

### Google Colab T4/A100 Setup
Colab instances benefit from the same cloning structure. Ensure your notebook is configured with a GPU backend:

1. Click on **Runtime** in the top menu.
2. Select **Change runtime type**.
3. Choose **T4 GPU** or **A100 GPU** under Hardware Accelerator.

Run the following cell to install Picotron and dependencies:

```bash
!git clone https://github.com/Syntropy-AI-Labs/picotron.git
%cd picotron
!pip install -e .
```

### Clearing Kernel Module Cache (Jupyter/Kaggle/Colab)
IPython kernels keep loaded package namespaces in memory. Always clear the cached imports prior to loading updated source files:

```python
import sys
for mod in list(sys.modules.keys()):
    if "picotron" in mod:
        del sys.modules[mod]
```

---

## 🖥️ Local Infrastructure Setup

For local machines with NVIDIA GPUs, we recommend using a clean Anaconda or Miniconda environment to manage dependencies.

### Workspace Initialization
```bash
# Create virtual environment
conda create -n picotron python=3.11 -y
conda activate picotron

# Install PyTorch with CUDA 12 support
pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# Clone and install in editable mode
git clone https://github.com/Syntropy-AI-Labs/picotron.git
cd picotron
pip install -e .
```

---

## ⚡ Verification Tests

To verify local GPU or multi-GPU environments, execute the included validation suites:

### Single GPU/CPU Smoke Test
```bash
python smoke_test.py
```

### Multi-GPU Distributed verification (torchrun)
```bash
torchrun --nproc_per_node=2 exclude/test_dual_gpu.py
```
