#! /bin/bash
#SBATCH -J ace
#SBATCH -p xxx
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH -o run_ace-%j.out
srun python3 -u train_ace.py --ted --split_prop 0.9 \
--data_path './tednet/artery_train_data' \
--model_save_path './tednet/artery_code/model/ace' \