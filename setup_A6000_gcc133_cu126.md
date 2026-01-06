TRELLIS Environment Setup Manual (RTX A6000 / CUDA 12.6 / GCC 13.3)

This manual is specifically designed for Ubuntu 24.04 systems using RTX A6000 (Ampere architecture) GPUs, compatible with CUDA 12.6 and PyTorch 2.8.0.

1. Basic Environment Initialization

# 1.1 Create and Activate Environment
conda create -n trellis python=3.10 -y
conda activate trellis

# 1.2 Install PyTorch 2.8.0 (Stable 2026 Version)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)

# 1.3 Set Environment Variables (Critical: A6000 Arch 8.6, Allow GCC 13)
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6"
export NVCC_FLAGS="-allow-unsupported-compiler"
export MAX_JOBS=8


2. Install Build Tools and General Dependencies

# 2.1 Basic Compilation Requirements
pip install ninja psutil "cython>=0.29.37" packaging setuptools

# 2.2 Gradio-Compatible Pillow Version
pip install "pillow<11.0" --no-deps


3. Compile and Install Core Operators (Critical: Disable Build Isolation)

For PyTorch 2.8.0, --no-build-isolation must be used to ensure the compiler utilizes the PyTorch version already installed in the environment.

# 3.1 Flash-Attention
pip install flash-attn --no-build-isolation

# 3.2 xFormers (v0.0.32 compatible with PyTorch 2.8)
pip install git+[https://github.com/facebookresearch/xformers.git@v0.0.32](https://github.com/facebookresearch/xformers.git@v0.0.32) --no-build-isolation


4. Install 3D Rendering Components

4.1 Kaolin (NVIDIA)

Use the precompiled wheel compatible with torch-2.8.0_cu126:

pip install kaolin==0.18.0 -f [https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu126.html](https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu126.html)


4.2 Nvdiffrast

pip install git+[https://github.com/NVlabs/nvdiffrast.git](https://github.com/NVlabs/nvdiffrast.git) --no-build-isolation


4.3 Diff-Gaussian-Rasterization (GCC 13 Patched Version)

Due to strict GCC 13.3 checks, the <cstdint> header must be manually inserted to fix uint32_t errors.

cd /tmp
git clone --recursive [https://github.com/graphdeco-inria/diff-gaussian-rasterization.git](https://github.com/graphdeco-inria/diff-gaussian-rasterization.git)
cd diff-gaussian-rasterization
# Automatic fix patch
sed -i '1i#include <cstdint>' cuda_rasterizer/rasterizer_impl.h
pip install . --no-build-isolation
cd ..


4.4 DiffOctreeRast (TRELLIS Exclusive)

git clone --recurse-submodules [https://github.com/JeffreyXiang/diffoctreerast.git](https://github.com/JeffreyXiang/diffoctreerast.git)
pip install ./diffoctreerast --no-build-isolation


5. Install Project Core Dependencies

Return to your project directory:

# Core Dependencies
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    rembg onnxruntime trimesh xatlas pyvista pymeshfix igraph transformers scipy \
    open3d spconv-cu120 gradio==4.44.1 gradio_litmodel3d==0.0.1

# Auxiliary 3D Utility Classes
pip install git+[https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8](https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8)


6. Running Guidelines

6.1 Resolve Insufficient VRAM and GPU ID Switching

If GPU 0 is occupied, specify an idle GPU (e.g., GPU 7) via CUDA_VISIBLE_DEVICES:

# Specify local cache path to avoid filling system disk
export HF_HOME=./cache
mkdir -p ./cache

# Run inference on GPU 7
CUDA_VISIBLE_DEVICES=7 HF_HOME=./cache python example.py


6.2 Quick Verification Script

Run this Python snippet to ensure all components are loaded correctly:

import torch
print(f"Device: {torch.cuda.get_device_name(0)}")
import flash_attn
import xformers
import kaolin
import nvdiffrast
import diffoctreerast
print("🎉 All components loaded successfully!")
