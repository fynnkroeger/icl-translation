#!/bin/bash
#SBATCH -A demelo-student
#SBATCH --partition sorcery
#SBATCH -C "ARCH:X86&GPU_MEM:40GB"
#SBATCH --gpus 1
#SBATCH --mem 32G
#SBATCH --cpus-per-task=16
#SBATCH --job-name "icl_translation_eval"
#SBATCH --output sbatch_out.txt
source "/hpi/fs00/home/fynn.kroeger/miniconda3/etc/profile.d/conda.sh"
conda activate alma-r
cd /hpi/fs00/home/fynn.kroeger/project/icl-translation
python inference.py
