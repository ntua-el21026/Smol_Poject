#!/bin/bash
#SBATCH --job-name=eval_stage1
#SBATCH --output=/scratch/%u/eval_logs/stage1/eval_stage1_%j.out
#SBATCH --error=/scratch/%u/eval_logs/stage1/eval_stage1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod

module purge
module load python/3.11
module load cuda/12.1
module load gcc/12
source /leonardo_scratch/large/userexternal/mpeppas0/venvs/nanotron_env/bin/activate

export EVAL_DATA_DIR=${EVAL_DATA_DIR:-/scratch/$USER/eval_data/stage1}
export EVAL_RESULTS_DIR=${EVAL_RESULTS_DIR:-/scratch/$USER/eval_results/stage1}
export EVAL_LOG_DIR=${EVAL_LOG_DIR:-/scratch/$USER/eval_logs/stage1}

mkdir -p "$EVAL_DATA_DIR" "$EVAL_RESULTS_DIR" "$EVAL_LOG_DIR"

CHECKPOINT_PATH=${CHECKPOINT_PATH:-$1}
STEP=${STEP:-$2}

if [ -z "$CHECKPOINT_PATH" ] || [ -z "$STEP" ]; then
  echo "Usage: sbatch --export=CHECKPOINT_PATH=...,STEP=... slurm/eval_stage1_every2ckpt.sbatch"
  exit 1
fi

python evaluation/run_eval_stage1.py --checkpoint_path "$CHECKPOINT_PATH" --step "$STEP"
