#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=unet_varray
#SBATCH --mail-type=END
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:1

module purge

pwd

nvidia-smi

singularity exec --nv --overlay ../overlay-10GB-400K.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c 'source /ext3/env.sh; python evaluate.py'
