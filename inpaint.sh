#!/bin/bash
data="celebahq"
sampleMethod="dpmsolver++"
type="dpmsolver"
steps="50"  # Maps to t_T_fine
steps_coarse="250"  # Maps to t_T
DIS="time_uniform"
order="3"
method="singlestep"
workdir="experiments/"$data"/"$steps_coarse"_"$steps"_"$DIS"_type-"$type

CUDA_VISIBLE_DEVICES=0 python main.py   --config $data".yml"   --exp=$workdir   --sample   --timesteps=$steps   --timesteps_coarse=$steps_coarse   --eta 0   --ni   --skip_type=$DIS   --sample_type=$sampleMethod   --dpm_solver_order=$order   --dpm_solver_method=$method   --dpm_solver_type=$type   --base_samples ./demo_split/celebahq/val   --mask_path ./demo_split/mask/thick   --use_inverse_masks False   --image_folder results/celeba/thick   --seed 1234   --port 12355
