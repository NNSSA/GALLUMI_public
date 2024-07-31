#!/bin/bash -l
#SBATCH --job-name=MTNG
#SBATCH --time=01-00:00:00
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=40000
#SBATCH --account=mkamion1

module load anaconda/2020.07
conda activate monty
python2 montepython/MontePython.py run --conf default.conf --output MTNG_clustering_minerror_9_20pctfixedcosmo -f 0.1 --param Clustering_MTNG_fixedcosmo.param --superupdate 20 --covmat MTNG_clustering_minerror_3/MTNG_clustering_minerror_3.covmat
