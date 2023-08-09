## How to train the model for predicting TMV and APTE/ACE

Before training, you need to convert CT data into a unified training format, please refer to `./artery_code/code_main/nii2npy.py`


### Train:
#### For TMV:
Generate gaussian density map:
```
python get_density_map.py --data_path './tednet/artery_train_data' --save_path './tednet/artery_code/artery_density_map' --num_thread 16
```

Visualization of density map:
```
python get_density_map.py --vis --vis_data_path './tednet/artery_density_map' --vis_save_path './tednet/artery_density_map_vis'
```

Train:
```
sbatch run_tmv.sh
sbatch run_ace.sh
sbatch run_roi.sh
sbatch run_roi_vein.sh
```
or
```
srun -p shlab_medical_pretrain --gres=gpu:1 --quotatype=auto \
python -u train_tmv.py --ted --split_prop 0.9 \
--data_path './tednet/artery_train_data' \
--density_path './tednet/artery_code/artery_density_map' \
--model_save_path './tednet/artery_code/model/tmv' \
```

If the number of samples is too large, you can train directly without generating density maps:
```
srun -p shlab_medical_pretrain --gres=gpu:1 --quotatype=auto \
python -u train_tmv.py --split_prop 0.9 \
--data_path './tednet/artery_train_data' \
--density_path './tednet/artery_code/artery_density_map' \
--model_save_path './tednet/artery_code/model/tmv' \
```

#### For ACE:
```
srun -p shlab_medical_pretrain --gres=gpu:1 --quotatype=auto \
python -u train_ace.py --ted --split_prop 0.9 \
--data_path './tednet/artery_train_data' \
--model_save_path './tednet/artery_code/model/ace' \
```

#### For ROI:
```
srun -p shlab_medical_pretrain --gres=gpu:1 --quotatype=auto \
python -u train_roi.py --split_prop 0.9 \
--data_path './tednet/artery_train_data' \
--model_save_path './tednet/artery_code/model/roi' \
```

### Test
#### For TMV [roi_path is optional]:
```
srun -p shlab_medical_pretrain --gres=gpu:1 --quotatype=spot \
python test_tmv.py --ted --data_path './tednet/artery_raw_data' \
--roi_path './tednet/artery_raw_data' \
--model_save_path './tednet/artery_model/tmv/model_best.pth' \
--result_save_path './tednet/results' \
--gt_label_path './tednet/artery_raw_data'
```

#### For ACE [roi_path is optional]:
```
srun -p shlab_medical_pretrain --gres=gpu:1 --quotatype=spot \
python test_ace.py --ted --data_path './tednet/artery_raw_data' \
--roi_path './tednet/artery_raw_data' \
--model_save_path './tednet/artery_model/ace/model_best.pth' \
--result_save_path './tednet/results' \
--gt_label_path './tednet/artery_raw_data'
```

