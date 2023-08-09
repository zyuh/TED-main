#! /bin/bash
#SBATCH -p xxx
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH --quotatype=spot

python -u vein_code/code_main/label_convert_test.py --data_path $1 --save_path $2 --num_thread 16

srun python -u vein_code/code_main/test_cap2d_fpn_aug.py --is_tumor_seg --is_hcccap --is_recover_rect --data_type cap --load_snapshot_path ./vein_code/model/cap_try_dcap_lr0.01_bs64_epo300_ncls4_joint/epoch_299.pth --dataset $1 --test_middle_save $2 --test_save $3 \

srun python -u vein_code/code_main/test_cap2d_fpn_aug.py --is_tumor_seg --is_hcccap --is_recover_rect --data_type fen --load_snapshot_path ./vein_code/model/cap_try_dfen_lr0.01_bs64_epo300_ncls4_joint/epoch_299.pth --dataset $1 --test_middle_save $2 --test_save $3 \

python -u vein_code/code_main/prepare4mvi_test.py --raw_data_path $1 --data_path $2 --save_path $3 --num_thread 16