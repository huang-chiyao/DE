#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: data_collection.sh [options]

Options:
  --shapenet-root PATH   Path to ShapeNetVox32 input tree
  --output-root PATH     Root for rendered outputs
  --split-root PATH      Root for split txt outputs (defaults to output root)
  --begin-idx N          Starting model index
  --end-idx N            Ending model index (exclusive)
  --num-images N         Images per model
  --num-processes N      Parallel worker count
  --mode MODE            render | split | all
  --split-strategy STR   view_level | model_level | balanced_model
  --blender-bin PATH     Blender executable
  -h, --help             Show this help

Example:
./data_collection.sh --shapenet-root ShapeNet/ShapeNetVox32/ --output-root output/
EOF
}

SHAPENET_ROOT_ARG=""
OUTPUT_ROOT_ARG=""
SPLIT_ROOT_ARG=""
BEGIN_IDX_ARG=""
END_IDX_ARG=""
NUM_IMAGES_ARG=""
NUM_PROCESSES_ARG=""
MODE_ARG=""
SPLIT_STRATEGY_ARG=""
BLENDER_BIN_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shapenet-root)
      SHAPENET_ROOT_ARG="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT_ARG="$2"
      shift 2
      ;;
    --split-root)
      SPLIT_ROOT_ARG="$2"
      shift 2
      ;;
    --begin-idx)
      BEGIN_IDX_ARG="$2"
      shift 2
      ;;
    --end-idx)
      END_IDX_ARG="$2"
      shift 2
      ;;
    --num-images)
      NUM_IMAGES_ARG="$2"
      shift 2
      ;;
    --num-processes)
      NUM_PROCESSES_ARG="$2"
      shift 2
      ;;
    --mode)
      MODE_ARG="$2"
      shift 2
      ;;
    --split-strategy)
      SPLIT_STRATEGY_ARG="$2"
      shift 2
      ;;
    --blender-bin)
      BLENDER_BIN_ARG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

echo "Loading software modules..."
module load mamba/latest
module load blender

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Changing to working directory: ${SCRIPT_DIR}"
cd "${SCRIPT_DIR}"

export PYTHONPATH="/home/ngocbach/.conda/envs/py3-blender/lib/python3.11/site-packages:$PYTHONPATH"
export BLENDER_PYTHON_PATH="$CONDA_PREFIX/bin/python"
export PYTHONPATH="$CONDA_PREFIX/lib/python3.11/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

CATEGORIES=(
  02691156
  02958343
  03001627
  04379243
  02828884
  03636649
  04090263
  04401088
  02933112
  04256520
)

BEGIN_IDX="${BEGIN_IDX_ARG:-${BEGIN_IDX:-0}}"
END_IDX="${END_IDX_ARG:-${END_IDX:-10}}"
NUM_IMAGES="${NUM_IMAGES_ARG:-${NUM_IMAGES:-300}}"
NUM_PROCESSES="${NUM_PROCESSES_ARG:-${NUM_PROCESSES:-16}}"
MODE="${MODE_ARG:-${MODE:-all}}"
SPLIT_STRATEGY="${SPLIT_STRATEGY_ARG:-${SPLIT_STRATEGY:-view_level}}"
BLENDER_BIN="${BLENDER_BIN_ARG:-${BLENDER_BIN:-blender}}"
SHAPENET_ROOT="${SHAPENET_ROOT_ARG:-${SHAPENET_ROOT:-${SCRIPT_DIR}/ShapeNet/ShapeNetVox32}}"
RENDER_ROOT="${OUTPUT_ROOT_ARG:-${RENDER_ROOT:-${SCRIPT_DIR}/ShapeNet/ShapeNetRendering_az_90_el_ro_45_cls_id_30k}}"
SPLIT_ROOT="${SPLIT_ROOT_ARG:-${SPLIT_ROOT:-${RENDER_ROOT}}}"

echo "ShapeNet root: ${SHAPENET_ROOT}"
echo "Render root: ${RENDER_ROOT}"
echo "Split root: ${SPLIT_ROOT}"

echo "Running unified dataset pipeline..."
python3 core/dataset_pipeline.py \
  --mode "${MODE}" \
  --categories "${CATEGORIES[@]}" \
  --begin-idx "${BEGIN_IDX}" \
  --end-idx "${END_IDX}" \
  --num-images "${NUM_IMAGES}" \
  --num-processes "${NUM_PROCESSES}" \
  --shapenet-root "${SHAPENET_ROOT}" \
  --render-root "${RENDER_ROOT}" \
  --split-output-root "${SPLIT_ROOT}" \
  --split-strategy "${SPLIT_STRATEGY}" \
  --blender-bin "${BLENDER_BIN}"
