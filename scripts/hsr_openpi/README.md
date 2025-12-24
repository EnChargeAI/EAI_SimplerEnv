
```bash
git clone https://github.com/airoa-org/hsr_openpi.git
cd hsr_openpi
git checkout remotes/origin/release
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT=../scripts/hsr_openpi/.venv uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ..

source $(pwd)/scripts/hsr_openpi/.venv/bin/activate
uv pip install -e . ".[torch]"


python scripts/hsr_openpi/evaluate_fractal.py --ckpt-path /data/checkpoints/pi0_fractal_low_mem_finetune2/my_experiment/17000
```