
```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT=../scripts/openpi/.venv uv sync
source ../scripts/openpi/.venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ..

source $(pwd)/scripts/openpi/.venv/bin/activate
uv pip install -e . ".[torch]"


huggingface-cli download --resume-download --repo-type model HaomingSong/openpi0-fractal-lora --local-dir /data/checkpoints/openpi0-fractal-lora

python scripts/openpi/evaluate_fractal2.py --ckpt-path /data/checkpoints/openpi0-fractal-lora
CUDA_VISIBLE_DEVICES=1 python scripts/openpi/evaluate_fractal.py --ckpt-path HaomingSong/openpi0-bridge-lora
```

