output_dir=$1

python -m torch.distributed.launch --nproc_per_node=4 --master_port=23450 \
  main.py -b ${output_dir}/config.yaml -t -r ${output_dir} -p mcd-vae-new  > ${output_dir}/out.log

