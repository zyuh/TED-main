## How to train the model for predicting CAP and FEN

Before training, you need to convert CT data into a unified training format, please refer to `./artery_code/code_main/nii2npy.py`

## Preprocessing
File Structure (for example):
```
./$vein_raw_data$
├── patient1_pid             # Name the folder with PID       
│   ├── $patient1_pid$_cap_5mm.nii
│   ├── $patient1_pid$_fen_5mm.nii
│   ├── $patient1_pid$_img_5mm.nii
│   └── $patient1_pid$_roi_5mm.nii      
├── patient2_pid                     
│   ├── $patient1_pid$_cap_5mm.nii
│   ├── $patient1_pid$_fen_5mm.nii
│   ├── $patient1_pid$_img_5mm.nii
│   └── $patient1_pid$_roi_5mm.nii      
```

Preprocessing:
```
python label_convert.py --data_path './tednet/vein_raw_data' --save_path './tednet/vein_train_data' --num_thread 16
```

Visualization
```
python label_convert.py --show --slice_path './tednet/vein_train_data/PH1469/PH1469_slice_35.npz' \
--vis_save_path './tednet/vein_code/vis_data'
```


Split dataset:
```
python split_dataset.py --data_path './tednet/vein_train_data' --save_path './tednet/vein_code/code_main/lists' --balance_type 'cap' --split_prop 0.9
python split_dataset.py --data_path './tednet/vein_train_data' --save_path './tednet/vein_code/code_main/lists' --balance_type 'fen' --split_prop 0.9
```


## Train
```
srun -p shlab_medical_pretrain --gres=gpu:1 --quotatype=auto \
python -u train_cap2d_fpn_aug.py --data_type cap \
--batch_size 64 --base_lr 0.01 --n_classes 4 --max_epochs 300 \
--is_recover_rect --is_tumor_seg --is_filteroutliers --is_hcccap --is_fasttrain \
--exp cap_try --list_path './tednet/vein_code/code_main/lists' \
--dataset './tednet/vein_train_data' \
```
```
srun -p shlab_medical_pretrain --gres=gpu:1 --quotatype=auto \
python -u train_cap2d_fpn_aug.py --data_type fen \
--batch_size 64 --base_lr 0.01 --n_classes 4 --max_epochs 300 \
--is_recover_rect --is_tumor_seg --is_filteroutliers --is_hcccap --is_fasttrain \
--exp cap_try --list_path './tednet/vein_code/code_main/lists' \
--dataset './tednet/vein_train_data' \
```

## Test
```
srun -p shlab_medical_pretrain --gres=gpu:1 python -u train_cap2d_fpn_aug.py \
--is_test --data_type cap \
--exp cap_try --is_tumor_seg --batch_size 64 --n_classes 4 --is_hcccap --is_recover_rect \
--list_path './tednet/vein_code/code_main/lists' \
--dataset './tednet/vein_train_data' \
--load_snapshot_path './tednet/vein_code/model/cap_try_dcap_lr0.01_bs64_epo300_ncls4_joint/epoch_299.pth' \
```
```
srun -p shlab_medical_pretrain --gres=gpu:1 python -u train_cap2d_fpn_aug.py \
--is_test --data_type fen \
--exp cap_try --is_tumor_seg --batch_size 64 --n_classes 4 --is_hcccap --is_recover_rect \
--list_path './tednet/vein_code/code_main/lists' \
--dataset './tednet/vein_train_data' \
--load_snapshot_path './tednet/vein_code/model/cap_try_dfen_lr0.01_bs64_epo300_ncls4_joint/epoch_299.pth' \
```


