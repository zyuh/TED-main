#! /bin/bash
#SBATCH -p xxx
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH --quotatype=spot

python -u train_mvi.py --data_path ./tednet/mvi_train_data --model_load_path mvi_code/model/model_best.pth --use_attr --use_ourtriplet --use_bio_marker