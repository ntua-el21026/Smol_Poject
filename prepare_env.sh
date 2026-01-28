#!/bin/bash

# Σταματάει το script αν υπάρξει οποιοδήποτε σφάλμα
set -e

echo "==================================================="
echo "   SETTING UP NANOTRON ENVIRONMENT FOR LEONARDO    "
echo "==================================================="

# 1. Φόρτωση System Modules (Leonardo Specifics)
# Χρειαζόμαστε CUDA και GCC για να γίνει compile το Flash Attention
echo "[1/6] Loading System Modules..."
module purge
module load python/3.11  # Ή 3.10 ανάλογα τι έχει default
module load cuda/12.1    # Απαραίτητο για τις A100
module load gcc/12       # Compiler για C++ extensions
module list

# 2. Δημιουργία Virtual Environment
echo "[2/6] Creating Virtual Environment (nanotron_env)..."
if [ -d "nanotron_env" ]; then
    echo "Environment 'nanotron_env' already exists. Updating it..."
else
    python -m venv nanotron_env
    echo "Created new environment."
fi

# Ενεργοποίηση
source nanotron_env/bin/activate
echo "Environment activated: $(which python)"

# 3. Upgrade pip (σημαντικό για binary wheels)
echo "[3/6] Upgrading pip..."
pip install --upgrade pip build wheel

# 4. PyTorch Installation (CUDA 12.1 Version)
echo "[4/6] Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Dependencies & Nanotron
echo "[5/6] Installing Dependencies & Nanotron..."
# Βασικά πακέτα
pip install numpy packaging pyyaml ninja tqdm wandb
# Hugging Face Ecosystem
pip install transformers datasets tokenizers
# Nanotron from Source
pip install git+https://github.com/huggingface/nanotron.git

# 6. Flash Attention 2 (Compiling from Source)
echo "[6/6] Installing Flash Attention 2..."
echo "WARNING: This step might take 5-10 minutes. Please wait..."
# Περιορίζουμε τα jobs για να μην κρασάρει το login node
export MAX_JOBS=4 
pip install flash-attn --no-build-isolation

echo "==================================================="
echo "             SETUP COMPLETED SUCCESSFULLY!         "
echo "==================================================="
echo "To activate the environment in the future, run:"
echo "source nanotron_env/bin/activate"
echo "==================================================="