#!/usr/bin/env bash
# Create .venv and install everything needed to run the app.
# All dependencies are listed in requirements/:
#   base.txt - GUI and data handling
#   sam2.txt - PyTorch and SAM2
#   dev.txt  - base + sam2 + tests/linters
set -e
cd "$(dirname "$0")/.."

# SAM2 sources come as a git submodule (third_party/sam2_upstream).
if [ ! -f third_party/sam2_upstream/setup.py ]; then
  git submodule update --init third_party/sam2_upstream
fi

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
SAM2_BUILD_CUDA=0 pip install -r requirements/dev.txt

echo "Done. Next: bash scripts/download_sam2_checkpoints.sh"
