FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 基本ツール
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=all \
    TZ=Asia/Tokyo \
    PYTHONUNBUFFERED=1 \
    VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion build-essential ca-certificates cmake curl git git-lfs \
    htop libegl1 libxext6 libjpeg-dev libpng-dev libvulkan1 rsync \
    tmux unzip vim wget xvfb pkg-config ffmpeg \
    libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglib2.0-0 \
    libsm6 libxrender1 libgomp1 libglu1-mesa libxi6 software-properties-common \
    python3.11 python3.11-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Python 3.11をデフォルトに設定
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Git LFSを有効化
RUN git lfs install

# Vulkan/EGL設定ファイルをコピー
COPY docker/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
COPY docker/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# 作業ディレクトリ
WORKDIR /workspace

# ソースコードをコピー（ビルド時のインストールに必要）
# 注意: 実行時は run_docker.sh でホストのコードがマウントされ、これは上書きされます
COPY . /workspace/

# SimplerEnv と ManiSkill2_real2sim をインストール
RUN cd /workspace/ManiSkill2_real2sim && pip install --no-cache-dir -e . && cd /workspace
RUN pip install --no-cache-dir -e /workspace
RUN pip install --no-cache-dir -r /workspace/requirements_full_install.txt

# PyTorch と torchvision (CUDA 12.1)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Flash Attention
RUN pip install --no-cache-dir --no-deps \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Isaac-GR00T のベース依存（--no-deps で重複を避ける）
RUN cd /workspace/Isaac-GR00T && pip install --no-cache-dir -e .[base] --no-deps && cd /workspace

# 追加ライブラリ（問題が発生しやすい依存関係を明示的にインストール）
RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    "pydantic>=2,<3" typing_extensions==4.12.2 --force-reinstall

RUN pip install --no-cache-dir \
    albumentations==1.4.18 albucore==0.0.17 scikit-image==0.25.2 lazy_loader==0.4 --no-deps

RUN pip install --no-cache-dir \
    decord==0.6.0 av==12.3.0 --no-deps

RUN pip install --no-cache-dir \
    nptyping==2.5.0 numpydantic==1.6.10 --no-deps

RUN pip install --no-cache-dir \
    diffusers==0.30.2 timm==1.0.14 peft==0.14.0

RUN pip install --no-cache-dir \
    transformers==4.51.3 --force-reinstall --no-deps

RUN pip install --no-cache-dir \
    pyzmq --no-deps

RUN pip install --no-cache-dir \
    "tokenizers>=0.21,<0.22" --no-deps

# PyTorch3D（ソースからビルド）
RUN pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git"

# NumPy < 2.0 を再度強制（PyTorch3Dがアップグレードする可能性があるため）
RUN pip install --no-cache-dir --force-reinstall "numpy>=1.24.4,<2.0"

# エントリーポイントスクリプトをコピー
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# エントリーポイントを設定（起動時に設定ファイルを自動コピー）
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
