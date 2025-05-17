#!/bin/bash
data="celebahq"
sampleMethod='dpmsolver++'
type="dpmsolver"
steps="50"
DIS="time_uniform"
order="3"
method="multistep"
workdir="experiments/"$data"/"$sampleMethod"_"$method"_order"$order"_"$steps"_"$DIS"_type-"$type

CUDA_VISIBLE_DEVICES=0 python main.py --config $data".yml" --exp=$workdir --sample --fid --timesteps=$steps --eta 0 --ni --skip_type=$DIS --sample_type=$sampleMethod --dpm_solver_order=$order --dpm_solver_method=$method --dpm_solver_type=$type --port 12350 