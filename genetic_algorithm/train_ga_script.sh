#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 1-12:00
#SBATCH -p debug
#SBATCH --mem=16000
#SBATCH -o myjob.o
#SBATCH -e myjob.e
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yg1227@nyu.edu
#SBATCH --constraint=2630v3
#SBATCH --gpu=gpu:0

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTEMPDIR
echo "working directory="$SLURM_SUBMIT_DIR

module load anaconda3/4.0.0 pytorch/cpu/0.4.0 python/gnu/3.5.1
echo "Launch train_ga.py with srun"
python train_ga.py InvertedPendulum-v2 --exp_name originalGA --gamma 0 --fitness_eval_episodes 30 --n_elite 20 --pop_size 50 -l 1 -s 32 -a tanh --output_activation tanh --n_generations 50 --sigma 0.05

