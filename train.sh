name=cylinder_asym_networks_sparse_conv_lstm
gpuid=0

CUDA_VISIBLE_DEVICES=${gpuid}  python3 train_cylinder_asym_cuda.py \
2>&1 | tee logs_dir/${name}_logs_tee.txt
