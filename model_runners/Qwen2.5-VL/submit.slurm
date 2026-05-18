#!/bin/bash
#

#SBATCH --job-name=VLMModel          # Job name
#SBATCH --output=log/OmniModel.%j.out     # Standard output log (%j is job ID)
#SBATCH --error=log/OmniModel.%j.err      # Standard error log
#SBATCH --partition=main              # Partition to submit to (adjust based on your system)
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4



module load conda
conda activate omni

# Training code
python generate_questions.py