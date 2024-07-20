GPUS=8
work_dir=work_dirs/mamba_fscil/cifar_resnet12_mambafscil
bash tools/dist_train.sh configs/cifar/resnet12_etf_bs512_200e_cifar_mambafscil.py $GPUS --work-dir ${work_dir} --seed 0 --deterministic
bash tools/run_fscil.sh configs/cifar/resnet12_etf_bs512_200e_cifar_eval_mambafscil.py ${work_dir} ${work_dir}/best.pth $GPUS --seed 0 --deterministic
