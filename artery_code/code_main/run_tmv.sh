#! /bin/bash
#SBATCH -J tmv
#SBATCH -p xxx
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH -o run_tmv-%j.out
srun python3 -u train_tmv.py --split_prop 0.9 \
--data_path './tednet/artery_train_data' \
--density_path './tednet/artery_code/artery_density_map' \
--model_save_path './tednet/artery_code/model/tmv' \