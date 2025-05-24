#!/bin/bash
data="celebahq"
sampleMethod="dpmsolver++"
type="dpmsolver"
steps="75"  # Maps to t_T_fine
steps_coarse="250"  # Maps to t_T
DIS="time_uniform"
order="3"
method="multistep"
workdir="experiments/"$data"/"$sampleMethod"_"$method"_order"$order"_"$steps"_"$DIS"_type-"$type

CUDA_VISIBLE_DEVICES=0 python main.py \
  --config $data".yml" \
  --exp=$workdir \
  --sample \
  --fid \
  --timesteps=$steps \
  --timesteps_coarse=$steps_coarse \
  --eta 0 \
  --ni \
  --skip_type=$DIS \
  --sample_type=$sampleMethod \
  --dpm_solver_order=$order \
  --dpm_solver_method=$method \
  --dpm_solver_type=$type \
  --base_samples ./demo/image \
  --mask_path ./demo/mask/thick \
  --use_inverse_masks False \
  --image_folder results/celebahq/thick \
  --seed 1234 \
  --port 12355