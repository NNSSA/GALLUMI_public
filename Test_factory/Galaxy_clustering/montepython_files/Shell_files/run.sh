#!/bin/bash -l
#SBATCH --job-name=MTNG
#SBATCH --time=02-20:00:00
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=5000
#SBATCH --account=mkamion1

module load anaconda/2020.07
conda activate monty
python2 montepython/MontePython.py run --conf default.conf --output MTNG_clustering_8_z4_1bin_10pct -f 0.2 --param Clustering_MTNG.param --superupdate 20 
