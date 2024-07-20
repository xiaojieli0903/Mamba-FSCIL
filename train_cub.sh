GPUS=8
work_dir=work_dirs/mamba_fscil/cub_resnet18_mambafscil
bash tools/dist_train.sh configs/cub/resnet18_etf_bs512_80e_cub_mambafscil.py $GPUS --work-dir ${work_dir} --seed 0 --deterministic
bash tools/run_fscil.sh configs/cub/resnet18_etf_bs512_80e_cub_eval_mambafscil.py ${work_dir} ${work_dir}/best.pth $GPUS --seed 0 --deterministic
