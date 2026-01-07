#!/bin/bash
# Script to fix PyTorch installation issues in trellis environment

set -e

echo "=== PyTorch Installation Fix Script ==="
echo ""

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Error: Conda environment not activated. Please run:"
    echo "  conda activate trellis"
    exit 1
fi

echo "Current environment: $CONDA_DEFAULT_ENV"
echo ""

# Check current PyTorch installation
echo "Checking current PyTorch installation..."
if python -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch can be imported"
    python -c "import torch; print('  Version:', torch.__version__)"
    python -c "import torch; print('  CUDA available:', torch.cuda.is_available())"
    if python -c "import torch" 2>/dev/null && python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
        python -c "import torch; print('  CUDA version:', torch.version.cuda)"
    fi
    exit 0
else
    echo "✗ PyTorch cannot be imported"
    echo ""
    echo "Attempting to fix..."
fi

# Option 1: Reinstall Intel MKL (common cause of iJIT_NotifyEvent error)
echo ""
echo "Step 1: Reinstalling Intel MKL libraries..."
# Try conda-forge first, then defaults
if ! conda install mkl mkl-service mkl_fft mkl_random -c conda-forge -y 2>/dev/null; then
    echo "Trying defaults channel..."
    conda install mkl mkl-service mkl_fft mkl_random -y
fi

# Option 2: Reinstall PyTorch
echo ""
echo "Step 2: Reinstalling PyTorch..."
# Detect CUDA version from system
if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "Detected CUDA version: $CUDA_VER"
    
    if [[ "$CUDA_VER" == "12.1"* ]]; then
        echo "Installing PyTorch 2.4.0 with CUDA 12.1..."
        conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
    elif [[ "$CUDA_VER" == "11.8"* ]]; then
        echo "Installing PyTorch 2.4.0 with CUDA 11.8..."
        conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        echo "Installing PyTorch 2.4.0 with CUDA 12.1 (default)..."
        conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
    fi
else
    echo "nvcc not found, installing PyTorch 2.4.0 with CUDA 12.1 (default)..."
    conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
fi

# Test installation
echo ""
echo "Step 3: Testing PyTorch installation..."
if python -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch successfully installed!"
    python -c "import torch; print('  Version:', torch.__version__)"
    python -c "import torch; print('  CUDA available:', torch.cuda.is_available())"
    if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
        python -c "import torch; print('  CUDA version:', torch.version.cuda)"
    fi
else
    echo "✗ PyTorch installation failed"
    echo ""
    echo "Please try creating a fresh environment:"
    echo "  conda create -n trellis python=3.10"
    echo "  conda activate trellis"
    echo "  conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia"
    exit 1
fi

echo ""
echo "=== Fix completed successfully! ==="

