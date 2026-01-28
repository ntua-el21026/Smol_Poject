#!/bin/bash
#SBATCH --job-name=prep-slimpajama
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32      
#SBATCH --time=06:00:00         
#SBATCH --partition=boost_usr_prod 
#SBATCH --mem=64GB              
#SBATCH --output=logs/prep_data_%j.out
#SBATCH --error=logs/prep_data_%j.err

source nanotron_env/bin/activate


echo "Starting data preparation on $(hostname)"
python scripts/prepare_data_stage1.py 