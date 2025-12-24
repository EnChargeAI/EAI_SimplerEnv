## Introduction

# SimplerEnvのMRを提出する際の注意事項

- MRを提出する際に、WASABIにアップされているモデル重みのパスを記載してください。
- 運営側でも並行してコード修正やバグ対応を進めております。そのため、MR提出日の時点で、各チームの提出ブランチに改めて `benchmark-v2` ブランチをマージしていただき、**Merge Conflictがない状態**にしていただけますよう、ご協力をお願いいたします。
- また、以下の評価スクリプトについては、各チームのモデルと統合した上で、**最後まで正常に実行できること**をご確認ください。

Google Robot Evaluation Script
```
CUDA_VISIBLE_DEVICES=<GPU_DEVICE_ID> python scripts/rt1/evaluate_fractal.py --ckpt-path /path/to/ckpt
```

WidowX Evaluation Script
```
CUDA_VISIBLE_DEVICES=<GPU_DEVICE_ID> python scripts/openpi/challenge_widowx.py --ckpt /path/to/ckpt 
```

### Google Robot Evaluation Task List

| Task  | Challenge | Task Definition                                     | Task                                | Randomizer Pool                                                                    |
| ----- | --------- | --------------------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------- |
| 1-1   | 1         | pick `<object>`                                     | `pick_object_visual_matching`       | `<object>`,`<position>`,`<object_orientation>`,`<robot_color>`,`<camera_position>` |
| 1-2   | 1         | pick `<object>`                                     | `pick_object_variant_agg`           | `<object>`,`<position>`,`<object_orientation>`,`<background/cabinet>`              |
| 2-1   | 1         | pick `<object>`                                     | `pick_object_among_visual_matching` | `<object>`,`<position>`,`<object_orientation>`,`<robot_color>`,`<camera_position>` |
| 2-2   | 1         | pick `<object>`                                     | `pick_object_among_variant_agg`     | `<object>`,`<position>`,`<object_orientation>`,`<background/cabinet>`              |
| 3-1   | 1         | open/close `<position>` drawer                      | `drawer_visual_matching`            | `<position>`,`<robot_color>`,`<background-robot_init_pos>`                         |
| 3-2   | 1         | open/close `<position>` drawer                      | `drawer_variant_agg`                | `<position>`,`<lighting>`,`<background>`,`<cabinet>`                               |
| 4-1   | 2         | move `<object>` near `<object>`                     | `move_near_visual_matching`         | `<object>`,`<position>`,`<robot_position>`,`<robot_color>`                         |
| 4-2   | 2         | move `<object>` near `<object>`                     | `move_near_variant_agg`             | `<object>`,`<position>`,`<lighting>`,`<background/cabinet>`,`<camera_position>`    |
| 5-1   | 2         | open top drawer -> place `<object>` into top drawer | `put_in_drawer_visual_matching`     | `<object>`,`<robot_color>`,`<background-robot_init_pos>`                           |
| 5-2   | 2         | open top drawer -> place `<object>` into top drawer | `put_in_drawer_variant_agg`         | `<object>`,`<lighting>`,`<robot_position>`,`<background>`,`<cabinet>`              |


### WidowX Evaluation Task List

| Task | Challenge | Task Definition               |
| ---- | --------- | ----------------------------- |
| 1    | 1         | Pick `<object>`               |
| 2    | 2         | Stacking `<cube>` on `<cube>` |
| 3    | 2         | Put `<object>` on `<top>`     |
| 4    | 2         | Put `<object>` in basket      |

**ランダマイザープール**

| Pool Name  | Element                                                    |
| ---------- | ---------------------------------------------------------- |
| `<object>` | `green cube`, `yellow cube`, `eggplant`, `spoon`, `carrot` |
| `<top>`    | `plate`, `towel`                                           |
| `<cube>`   | `green cube`, `yellow cube`                                |


## 環境構築

詳細なそれぞれの環境のドキュメント
- [RT-1](scripts/rt1/README.md)
- [OpenPi](scripts/openpi/README.md)
- [lerobot-pi0](scripts/lerobotpi/README.md)

### 1. Docker構築
```bash
git clone https://github.com/airoa-org/SimplerEnv.git
cd SimplerEnv
git checkout origin/benchmark-v2
git submodule update --init --recursive
cp docker/Dockerfile .
docker build -t simpler_env .
```

### 2. モデルインストール～実行

#### RT-1

インストール
```bash
# Install Google Cloud SDK
cd ..
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
cd SimplerEnv

# Create a virtual environment
uv venv -p 3.10 scripts/rt1/.venv

# Install dependencies
source $(pwd)/scripts/rt1/.venv/bin/activate
uv pip install tensorflow==2.15.0
uv pip install -r requirements_full_install.txt
uv pip install -e .
uv pip install tensorflow[and-cuda]==2.15.1

# If you encounter an import error.
# ImportError: This version of TensorFlow Probability requires TensorFlow version >= 2.16; Detected an installation of version 2.15.1. Please upgrade TensorFlow to proceed.
# Do the following.
uv pip uninstall tensorflow-probability
uv pip install "tensorflow-probability==0.22.1" --no-deps


# Make a checkpoint dir:
mkdir checkpoints

# RT-1-X
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip .
unzip rt_1_x_tf_trained_for_002272480_step.zip
mv rt_1_x_tf_trained_for_002272480_step checkpoints
rm rt_1_x_tf_trained_for_002272480_step.zip

# RT-1-Converged
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_tf_trained_for_000400120 .
mv rt_1_tf_trained_for_000400120 checkpoints

# RT-1-15%
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_tf_trained_for_000058240 .
mv rt_1_tf_trained_for_000058240 checkpoints

# RT-1-Begin
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_tf_trained_for_000001120 .
mv rt_1_tf_trained_for_000001120 checkpoints      
```

実行
```bash
# fractal
CUDA_VISIBLE_DEVICES=1 python scripts/rt1/evaluate_fractal.py --ckpt-path checkpoints/rt_1_tf_trained_for_000400120
```

#### OpenPi

インストール
```bash
git clone your_openpi
# サブモジュールの中にもbranchの切り替えとかが必要の可能性がある
# 例
# git branch -a
# git checkout remotes/origin/feature/fractal
cd your_openpi
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT=../scripts/openpi/.venv uv sync
source ../scripts/openpi/.venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ..

source $(pwd)/scripts/openpi/.venv/bin/activate
uv pip install -e . ".[torch]"

huggingface-cli download --resume-download --repo-type model HaomingSong/openpi0-fractal-lora --local-dir /path/to/ckpt
```

実行
1. OpenPiの場合モデルを起動
```bash
# current directory: /app/SimplerEnV
cd openpi
export SERVER_ARGS="policy:checkpoint --policy.config=pi0_bridge_low_mem_finetune --policy.dir=/path/to/ckpt"
uv run scripts/serve_policy.py $SERVER_ARGS
```

2. 実行
```bash
# current directory: /app/SimplerEnV
CUDA_VISIBLE_DEVICES=1 python scripts/openpi/challenge_widowx.py --ckpt /path/to/ckpt --control-freq 5
```

#### lerobot-pi0

インストール
```bash
uv venv -p 3.10 scripts/lerobotpi/.venv
source $(pwd)/scripts/lerobotpi/.venv/bin/activate

uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install git+https://github.com/huggingface/lerobot.git@ce2b9724bfe1b5a4c45e61b1890eef3f5ab0909c#egg=lerobot[pi0]
uv pip install -e . ".[torch]"
uv pip install pytest
```

実行
```bash
huggingface-cli login
CUDA_VISIBLE_DEVICES=1 python scripts/lerobotpi/evaluate_fractal.py --ckpt-path HaomingSong/lerobot-pi0-fractal
CUDA_VISIBLE_DEVICES=1 python scripts/lerobotpi/evaluate_fractal.py --ckpt-path lerobot/pi0
```

