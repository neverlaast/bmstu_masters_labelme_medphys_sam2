#!/usr/bin/env bash
# Download SAM2.1 checkpoints from the official Meta server and verify SHA-256.
#
#   bash scripts/download_sam2_checkpoints.sh            # small (worker default)
#   bash scripts/download_sam2_checkpoints.sh tiny large
#   bash scripts/download_sam2_checkpoints.sh all
#
# Files go to third_party/sam2_upstream/, where the worker looks for them.
# URLs: third_party/sam2_upstream/checkpoints/download_ckpts.sh (SAM 2.1, 2024-09-28).
# Checksums were recorded from the checkpoints this project was developed and
# tested with; a mismatch means a corrupted or different file.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
TARGET="third_party/sam2_upstream"

sha_for() {
  case "$1" in
    tiny)      echo 7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69 ;;
    small)     echo 6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38 ;;
    base_plus) echo a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5 ;;
    large)     echo 2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318 ;;
    *) return 1 ;;
  esac
}

sha256() {
  if command -v shasum >/dev/null; then shasum -a 256 "$1" | cut -d' ' -f1
  else sha256sum "$1" | cut -d' ' -f1; fi
}

models=("$@")
[ ${#models[@]} -eq 0 ] && models=(small)
[ "${models[0]}" = all ] && models=(tiny small base_plus large)

[ -d "$TARGET" ] || { echo "Missing $TARGET (clone SAM2 first, see README)" >&2; exit 1; }

for model in "${models[@]}"; do
  expected=$(sha_for "$model") || { echo "Unknown model '$model' (tiny|small|base_plus|large|all)" >&2; exit 2; }
  file="$TARGET/sam2.1_hiera_${model}.pt"
  if [ -f "$file" ] && [ "$(sha256 "$file")" = "$expected" ]; then
    echo "OK      $file (already present)"
    continue
  fi
  echo "Download sam2.1_hiera_${model}.pt"
  partial="$file.part"
  # -C - resumes an interrupted download; the file is published only after verification.
  curl -fL --retry 3 -C - -o "$partial" "$BASE_URL/sam2.1_hiera_${model}.pt"
  actual=$(sha256 "$partial")
  if [ "$actual" != "$expected" ]; then
    rm -f "$partial"
    echo "SHA-256 mismatch for $model: $actual (expected $expected)" >&2
    exit 3
  fi
  mv "$partial" "$file"
  echo "OK      $file"
done
