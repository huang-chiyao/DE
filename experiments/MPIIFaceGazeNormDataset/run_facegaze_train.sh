#!/bin/bash
#SBATCH -A grp_ychoi131
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH -G a100:1       # Request 1 GPU (A100)
#SBATCH -p general
#SBATCH -q public
#SBATCH -o /scratch/ngocbach/tmp/domain/slurm_logs/ShapeNet/domain/slurm_%j_bc.out
#SBATCH -e /scratch/ngocbach/tmp/domain/slurm_logs/ShapeNet/domain/slurm_%j_bc.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngocbach@asu.edu
#SBATCH --export=NONE

# Usage: ./run_mix_shapenet_train.sh [--config CONFIG_FILE] [additional args...]
# Examples:
#   sbatch run_mix_shapenet_train.sh
#   sbatch run_mix_shapenet_train.sh --config debug
#   sbatch run_mix_shapenet_train.sh --debug --epochs 100
#   sbatch run_mix_shapenet_train.sh --config debug --debug --temp 0.5

set -euo pipefail

# Get script directory
SCRIPT_DIR="/scratch/ngocbach/tmp/domain/experiments/MPIIFaceGazeNormDataset"
CONFIG_DIR="${SCRIPT_DIR}/config"

# Default configuration
CONFIG_FILE="${CONFIG_DIR}/default.conf"
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            if [[ "$2" == *".conf" ]]; then
                CONFIG_FILE="$2"
            else
                CONFIG_FILE="${CONFIG_DIR}/${2}.conf"
            fi
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--config CONFIG_FILE] [additional args...]"
            echo ""
            echo "Options:"
            echo "  --config CONFIG     Use config file (default, debug, or path to .conf file)"
            echo "  [additional args]   Any additional arguments to pass to Python script"
            echo ""
            echo "Examples:"
            echo "  sbatch $0"
            echo "  sbatch $0 --config debug"
            echo "  sbatch $0 --debug --epochs 100"
            echo "  sbatch $0 --config debug --debug --temp 0.5"
            echo ""
            echo "Available config files:"
            ls -1 "${CONFIG_DIR}"/*.conf 2>/dev/null | sed 's/.*\//  /' | sed 's/\.conf$//' || echo "  No config files found"
            exit 0
            ;;
        *)
            # All other arguments are passed through to Python
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Load configuration
if [[ -f "$CONFIG_FILE" ]]; then
    echo "Loading configuration from: $CONFIG_FILE"
    source "$CONFIG_FILE"
else
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Display configuration
echo "=== EXPERIMENT CONFIGURATION ==="
echo "Config file: $CONFIG_FILE"
echo "Base arguments: $BASE_ARGS"
echo "Extra arguments: ${EXTRA_ARGS[*]}"
echo "================================="

# Run the experiment
module load mamba/latest
source activate vocal
wandb login --verify
cd "$BASE_DIR"

echo "Parsing method from arguments..."
METHOD="Domain"  # default

for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
    if [[ "${EXTRA_ARGS[$i]}" == "--method" && $((i+1)) -lt ${#EXTRA_ARGS[@]} ]]; then
        METHOD="${EXTRA_ARGS[$((i+1))]}"
        break
    fi
done

case "$METHOD" in
    Domain)
        MAIN_SCRIPT="main_mix.py"
        ;;
    RNC)
        MAIN_SCRIPT="main_mix.py"
        ;;
    L1)
        MAIN_SCRIPT="main_l1.py"
        ;;
    *)
        echo "Unknown method: $METHOD"
        exit 1
        ;;
esac

echo "Starting training with $MAIN_SCRIPT..."
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    eval "python $MAIN_SCRIPT $BASE_ARGS ${EXTRA_ARGS[*]}"
else
    eval "python $MAIN_SCRIPT $BASE_ARGS"
fi
