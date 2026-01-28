#!/bin/bash
#SBATCH --job-name=build_environment
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1           # Ζητάμε 1 GPU για να βρει τους drivers
#SBATCH --cpus-per-task=16     # CPUs για γρήγορο compile
#SBATCH --time=01:00:00        # 1 ώρα είναι υπεραρκετή
#SBATCH --partition=boost_usr_prod
#SBATCH --output=logs/build_env_%j.out
#SBATCH --error=logs/build_env_%j.err

echo "Starting Environment Build on $(hostname)"

# Τρέχουμε το script που φτιάξαμε
./prepare_env.sh

echo "Build Job Finished."