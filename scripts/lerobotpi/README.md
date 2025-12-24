
```bash
uv venv -p 3.10 scripts/lerobotpi/.venv


# uv pip install -e .
source $(pwd)/scripts/lerobotpi/.venv/bin/activate
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install git+https://github.com/huggingface/lerobot.git@ce2b9724bfe1b5a4c45e61b1890eef3f5ab0909c#egg=lerobot[pi0]
uv pip install -e . ".[torch]"
uv pip install pytest

huggingface-cli login
CUDA_VISIBLE_DEVICES=1 python scripts/lerobotpi/evaluate_fractal.py --ckpt-path HaomingSong/lerobot-pi0-fractal
CUDA_VISIBLE_DEVICES=1 python scripts/lerobotpi/evaluate_fractal.py --ckpt-path lerobot/pi0
```