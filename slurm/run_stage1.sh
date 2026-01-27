#!/bin/bash
#SBATCH --job-name=smol_stage1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00        # Αρκετός χρόνος για 10B tokens
#SBATCH --partition=boost_usr_prod
#SBATCH --account=<YOUR_ACCOUNT_NAME> # Π.χ. euinfra04...
#SBATCH --qos=normal

# --- 1. Environment Setup ---
echo "--- [1] Setting up Environment ---"
module load nvhpc/23.1
module load python/3.10 # Ή όποιο module χρησιμοποιείς
source /leonardo_scratch/large/userexternal/mpeppas0/venvs/nanotron_env/bin/activate # Το venv σου

# WandB Logging (Απαραίτητο για monitoring)
export WANDB_API_KEY=...
export WANDB_PROJECT="pretrain_stage1"
export WANDB_NAME="stage1-run-$(date +%Y%m%d_%H%M)"
export WANDB_MODE="offline"

# Optimization Flags
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO

# --- 2. Smol Fix: Scratch -> NVMe Strategy ---
# Αντιγράφουμε τα δεδομένα στον τοπικό δίσκο του Node ($TMPDIR) για να μην "μπουκώσει" το δίκτυο.
echo "--- [2] Moving Data to Local NVMe ($TMPDIR) ---"

# Ορίζουμε πού είναι τα data στο Scratch
SOURCE_DATA_PART1="/leonardo_scratch/large/userexternal/mpeppas0/dataset/slimpajama_part1"
SOURCE_DATA_PART2="/leonardo_scratch/large/userexternal/mpeppas0/dataset/slimpajama_part2"

# Ορίζουμε πού θα πάνε στο Node (Αυτό πρέπει να ταιριάζει με το config.yaml!)
DEST_DATA="/tmp/slimpajama_local" 

mkdir -p $DEST_DATA

# Μέτρηση χρόνου αντιγραφής
start_time=$(date +%s)

echo "Copying Part 1..."
cp -r $SOURCE_DATA_PART1/* $DEST_DATA/
echo "Copying Part 2..."
cp -r $SOURCE_DATA_PART2/* $DEST_DATA/

end_time=$(date +%s)
echo "Data copy finished in $((end_time - start_time)) seconds."

# --- 3. Checkpoint Logic (Auto-Resume) ---
# Αν υπάρχει checkpoint, το Nanotron το βρίσκει μόνο του αρκεί το config να δείχνει στο σωστό path.
# Εμείς απλά βεβαιωνόμαστε ότι το output folder υπάρχει.
CHECKPOINT_PATH="/leonardo_scratch/large/userexternal/mpeppas0/checkpoints/stage1"
mkdir -p $CHECKPOINT_PATH

# --- 4. Launch Training ---
echo "--- [3] Launching Nanotron Training ---"

# Χρησιμοποιούμε torchrun για 4 GPUs
torchrun --nproc_per_node=4 run_train.py --config-file config/config_stage1.yaml

