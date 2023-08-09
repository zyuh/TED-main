#!/bin/bash
data_path="./tednet/test_raw_data"
middle_path="./tednet/test_intermediate_data"
results_path="./tednet/test_results"
sbatch ./tednet/artery_code/code_main/run_artery.sh ${data_path} ${middle_path} ${results_path}
sbatch ./tednet/vein_code/code_main/run_vein.sh ${data_path} ${middle_path} ${results_path}
sbatch ./tednet/mvi_code/code_main/run_mvi.sh ${data_path} ${middle_path} ${results_path}
