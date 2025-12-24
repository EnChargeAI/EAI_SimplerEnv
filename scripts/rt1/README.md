
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


# Evaluate
CUDA_VISIBLE_DEVICES=3 python scripts/rt1/evaluate_fractal.py --ckpt-path checkpoints/rt_1_tf_trained_for_000400120
```