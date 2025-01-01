#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=sr
#SBATCH --error=test_error_train.log
#SBATCH --output=test_output_train.log
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jalajlochan@kgpian.iitkgp.ac.in
module apps/lammps/29Oct2020/cuda10.2/gpu
touch test_error_train.log
touch test_output_train.log
conda init
conda activate sr
CUDA_VISIBLE_DEVICES=0 python3 training.py -t ./images/splits/split.txt -s ./save -b 1 -m 0 --model ./RodoSol_pretrained_model.pt
