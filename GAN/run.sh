#!/bin/bash -l

#
# > man sbatch
#
# Nom du job 
#SBATCH --job-name=umbsdgan
#
# Fichier de sortie d'exécution
#SBATCH --output=training_umbsdgan.log

 
conda activate python37
setcuda 10.2

python3 GAN/U_MBSD_GAN.py --n_epochs 1000 --batch_size 64 --lr 0.00005 --latent_dim 100 --n_critic 5 --k 1 --reg 100