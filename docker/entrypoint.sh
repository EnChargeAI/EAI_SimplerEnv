#!/bin/bash
set -e

# Vulkan設定（レンダリングに必要）
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

# GR00T設定ファイルを上書き（起動時に毎回実行）
echo "Copying GR00T configuration files..."
cp /workspace/scripts/gr00t/data_config.py /workspace/Isaac-GR00T/gr00t/experiment/data_config.py
cp /workspace/scripts/gr00t/embodiment_tags.py /workspace/Isaac-GR00T/gr00t/data/embodiment_tags.py
echo "Configuration files copied successfully."

# 渡されたコマンドを実行
exec "$@"
