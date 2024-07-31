#!/bin/bash -l
#SBATCH --job-name=MTNG
#SBATCH --time=02-20:00:00
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=20000
#SBATCH --account=mkamion1

module load anaconda/2020.07
conda activate monty
python2 montepython/MontePython.py run --conf default.conf --output MTNG_clustering_fixedz4_new_4_5bins -f 0.3 --param Clustering_MTNG_fixedz.param --superupdate 20 --bestfit MTNG_clustering_fixedz4_new_1/MTNG_clustering_fixedz4_new_1.bestfit --covmat MTNG_clustering_fixedz4_new_1/MTNG_clustering_fixedz4_new_1.covmat
