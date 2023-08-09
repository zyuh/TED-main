#! /bin/bash
#SBATCH -p xxx
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH --quotatype=spot

srun python -u artery_code/code_main/test_roi.py --data_path $1 --result_save_path $2 --model_save_path artery_code/model/roi/model_best.pth --use_gaussian --do_tta

srun python -u artery_code/code_main/test_tmv.py --data_path $1 --result_save_path $2 --model_save_path artery_code/model/tmv/model_best.pth --use_gaussian --do_tta

srun python -u artery_code/code_main/test_ace.py --data_path $1 --result_save_path $2 --model_save_path artery_code/model/ace/model_best.pth --use_gaussian --do_tta --ted

srun python -u artery_code/code_main/test_vein_roi.py --data_path $1 --result_save_path $2 --model_save_path artery_code/model/vein_roi/model_best.pth --use_gaussian --do_tta
