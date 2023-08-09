#! /bin/bash
#SBATCH -J roi
#SBATCH -p xxx
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH -o run_vein_roi-%j.out
srun python3 -u train_vein_roi.py --split_prop 0.9 \
--data_path './tednet/vein_raw_data' \
--model_save_path './tednet/artery_code/model/vein_roi' \