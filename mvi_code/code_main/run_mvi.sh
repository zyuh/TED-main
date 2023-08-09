#! /bin/bash
#SBATCH -p xxx
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH --quotatype=spot

python -u mvi_code/code_main/test_mvi.py --data_path ./tednet/test_results --model_load_path ./tednet/mvi_code/model/model_best.pth --use_attr --use_ourtriplet --use_bio_marker