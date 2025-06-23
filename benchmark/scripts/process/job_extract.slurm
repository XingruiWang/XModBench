#!/bin/bash
#SBATCH --job-name=VLM_eval        
#SBATCH --output=log/job_%j.out
#SBATCH --error=log/job_%j.log    
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=32 

module load conda
conda activate omni

python generate_solos.py
