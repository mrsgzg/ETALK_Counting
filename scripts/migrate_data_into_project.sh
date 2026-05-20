#!/usr/bin/env bash
set -euo pipefail

# Copy external dataset and CSV files into this repository's standard layout:
#   data/ball_data_collection
#   data/Tools_script
#
# Usage:
#   bash scripts/migrate_data_into_project.sh
#   bash scripts/migrate_data_into_project.sh /path/to/ball_data_collection /path/to/Tools_script

SRC_DATA_ROOT="${1:-/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/ball_data_collection}"
SRC_TOOLS_ROOT="${2:-/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Ball_counting_CNN/Tools_script}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST_DATA_ROOT="${PROJECT_ROOT}/data/ball_data_collection"
DST_TOOLS_ROOT="${PROJECT_ROOT}/data/Tools_script"

echo "[1/4] Checking source paths..."
if [[ ! -d "${SRC_DATA_ROOT}" ]]; then
  echo "Error: source data directory not found: ${SRC_DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${SRC_TOOLS_ROOT}" ]]; then
  echo "Error: source tools directory not found: ${SRC_TOOLS_ROOT}" >&2
  exit 1
fi

echo "[2/4] Preparing destination directories..."
mkdir -p "${DST_DATA_ROOT}" "${DST_TOOLS_ROOT}"

echo "[3/4] Copying dataset (this may take a while)..."
if command -v rsync >/dev/null 2>&1; then
  rsync -a --info=progress2 "${SRC_DATA_ROOT}/" "${DST_DATA_ROOT}/"
else
  cp -a "${SRC_DATA_ROOT}/." "${DST_DATA_ROOT}/"
fi

echo "[4/4] Copying CSV files..."
if command -v rsync >/dev/null 2>&1; then
  rsync -a "${SRC_TOOLS_ROOT}/" "${DST_TOOLS_ROOT}/"
else
  cp -a "${SRC_TOOLS_ROOT}/." "${DST_TOOLS_ROOT}/"
fi

echo "Done. Data has been migrated into:"
echo "  ${DST_DATA_ROOT}"
echo "  ${DST_TOOLS_ROOT}"

echo "Next step: run training scripts from project root with default arguments or pass explicit paths if needed."
