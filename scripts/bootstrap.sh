#!/usr/bin/env bash
# Create .venv and install everything needed to run the app on a clean machine.
set -e
cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements/dev.txt
pip install torch torchvision

# SAM2 sources come as a git submodule (third_party/sam2_upstream).
if [ ! -f third_party/sam2_upstream/setup.py ]; then
  git submodule update --init third_party/sam2_upstream
fi
SAM2_BUILD_CUDA=0 pip install -e third_party/sam2_upstream

echo "Done. Next: bash scripts/download_sam2_checkpoints.sh"
