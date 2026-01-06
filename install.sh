# # 1. 创建并激活环境
# conda create -n trellis python=3.10 -y
# conda activate trellis

# 2. 安装 PyTorch 2.8.0 (CUDA 12.6)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# 3. 配置环境变量 (针对 A6000 优化)
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# [关键] A6000 是 Ampere 架构，算力 8.6
export TORCH_CUDA_ARCH_LIST="8.6"

# [关键] 防止 GCC 13.3 太新导致 nvcc 报错 (Optional, 但推荐加上)
export NVCC_FLAGS="-allow-unsupported-compiler"

# 4. 安装基础构建工具
pip install ninja setuptools wheel packaging
conda install -c conda-forge 'libstdcxx-ng>=11.2.0' -y

# 5. 克隆 TRELLIS 并安装 Python 依赖
# git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd TRELLIS

# 安装通用依赖
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    rembg onnxruntime trimesh xatlas pyvista pymeshfix igraph transformers scipy

# 安装 utils3d
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# 6. 编译核心算子 (Flash-Attn & xFormers)
# 显式指定架构后，这里编译会快很多
pip install xformers --no-binary xformers
pip install flash-attn --no-build-isolation

# 7. 编译 Kaolin (必须源码编译)
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
export IGNORE_TORCH_VER=1
pip install -r tools/requirements.txt
# 下面这步会消耗一些时间，请耐心等待
python setup.py install
cd ..

# 8. 编译图形学扩展
# Nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast.git

# Diff Octree Rast (Trellis 核心组件)
# 如果本地目录存在直接安装，不存在则尝试在线拉取(假设有公开源)
if [ -d "extensions/diffoctreerast" ]; then
    pip install ./extensions/diffoctreerast
else
    echo "Check extensions/diffoctreerast folder inside TRELLIS root."
fi

# Gaussian Rasterization
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git

# 9. 安装 Spconv
# 依然推荐 cu120 版本兼容 CUDA 12.x 系列
pip install spconv-cu120

# 10. 其他应用层工具
pip install open3d
pip install gradio==4.44.1 
pip install gradio_litmodel3d==0.0.1