# SimplerEnv Docker Environment

このディレクトリには、SimplerEnv (benchmark-v2) のDocker環境を構築するためのスクリプトが含まれています。

## 前提条件

- NVIDIA Driver >= 535 / CUDA 12.1 相当
- Docker がインストールされていること
- nvidia-docker (NVIDIA Container Toolkit) がインストールされていること
- Linux + GPU ノード

## クイックスタート

### 1. Docker イメージをビルド

リポジトリのルートディレクトリで以下を実行します。

```bash
bash docker/build_docker.sh
```

ビルドには30〜60分程度かかります。このプロセスでは以下がインストールされます：

- Python 3.11
- PyTorch 2.5.1 (CUDA 12.1)
- SimplerEnv と ManiSkill2_real2sim
- Isaac-GR00T と全ての依存関係
- Flash Attention 2.7.1
- PyTorch3D
- その他の必要なライブラリ

### 2. コンテナを起動

```bash
bash docker/run_docker.sh
```

### 3. コンテナに入る

```bash
docker exec -it simplerenv bash
```

## 使用方法

コンテナ内では、すべての依存関係がインストール済みなので、そのまま評価スクリプトを実行できます。

### WidowX (Bridge) での評価

```bash
python scripts/gr00t/evaluate_bridge.py \
  --ckpt-path /path/to/checkpoint-group6/
```

### Google Robot (Fractal) での評価

```bash
python scripts/gr00t/evaluate_fractal.py \
  --ckpt-path /path/to/checkpoint-group6/
```

## 技術詳細

### インストールされる主要なパッケージ

- **PyTorch**: 2.5.1 (CUDA 12.1)
- **Flash Attention**: 2.7.1.post4
- **Transformers**: 4.51.3
- **Diffusers**: 0.30.2
- **Timm**: 1.0.14
- **PEFT**: 0.14.0
- **PyTorch3D**: latest from source
- **その他**: pandas, pydantic, albumentations, decord, av, nptyping, numpydantic, pyzmq, tokenizers

### Dockerfileの構成

1. ベースイメージ: `nvidia/cuda:12.1.0-devel-ubuntu22.04`
2. システムパッケージのインストール
3. Python 3.11のセットアップ
4. SimplerEnv と ManiSkill2_real2sim のインストール
5. PyTorch とその他のディープラーニングライブラリのインストール
6. Isaac-GR00T の依存関係のインストール
7. 設定ファイルの上書き