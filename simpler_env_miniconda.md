# SimplerEnv: Setup & Evaluation (group6/benchmark-v2)

## 0. 前提

* NVIDIA Driver ≥ 535 / CUDA 12.1 相当
* Linux + GPU ノード
* Git LFS が使えること（初回だけ）:

  ```bash
  conda install -c conda-forge -y git-lfs && git lfs install
  ```

## 1. 取得 & ブランチ切替（サブモジュール込み）

```bash
git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules
cd SimplerEnv

# 提出ブランチへ（compare: group6/benchmark-v2 → base: benchmark-v2）
git checkout group6/benchmark-v2

# 念のためサブモジュールを初期化
git submodule update --init --recursive
```

## 2. Conda 環境

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n simpler_env_group6 -c conda-forge python=3.11
conda activate simpler_env_group6
```

## 3. 依存インストール（SimperEnv / ManiSkill / System）

```bash
# (A) SimplerEnv / ManiSkill2_real2sim（編集可能インストール）
cd ManiSkill2_real2sim && pip install -e . && cd ..
pip install -e .
pip install -r requirements_full_install.txt

# (B) System系（Vulkan, ffmpeg, gsutil）
conda install -y -c conda-forge vulkan-tools vulkan-headers ffmpeg=4.2.2 gsutil
```

## 4. PyTorch / Flash-Attn（CUDA 12.1）

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --no-cache-dir

pip install --no-deps \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/\
flash_attn-2.7.1.post4+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

## 5. Isaac-GR00T（サブモジュール、推論に必要な最小セット）

```bash
cd Isaac-GR00T

# ベースの依存（重い依存は上で入れているため --no-deps 併用）
pip install -e .[base] --no-deps

# 追加ライブラリ（必要だったものをピン留め）
pip install \
  pandas==2.2.3 \
  "pydantic>=2,<3" typing_extensions==4.12.2 --force-reinstall \
  albumentations==1.4.18 albucore==0.0.17 scikit-image==0.25.2 lazy_loader==0.4 --no-deps \
  decord==0.6.0 av==12.3.0 --no-deps \
  nptyping==2.5.0 numpydantic==1.6.10 --no-deps \
  diffusers==0.30.2 timm==1.0.14 peft==0.14.0 \
  transformers==4.51.3 --force-reinstall --no-deps \
  pyzmq --no-deps \
  "tokenizers>=0.21,<0.22" --no-deps \
  "git+https://github.com/facebookresearch/pytorch3d.git"

# サブモジュールのGR00T側の設定を上書き
cd ..
cp ./scripts/gr00t/data_config.py ./Isaac-GR00T/gr00t/experiment/data_config.py
cp ./scripts/gr00t/embodiment_tags.py ./Isaac-GR00T/gr00t/data/embodiment_tags.py
```

> 余計な依存衝突を避けるため、意図的に一部 `--no-deps` を使っています。
> 追加で必要な依存が出た場合は、そのライブラリのみ都度入れてください。

## 6. 推論テスト

WidowX（Bridge）:

```bash
python scripts/gr00t/evaluate_bridge.py \
  --ckpt-path /hogehoge-wasabi/path/to/checkpoint-group6/
```

Google Robot（Fractal）:

```bash
python scripts/gr00t/evaluate_fractal.py \
  --ckpt-path /hogehoge-wasabi/path/to/checkpoint-group6/
```
---