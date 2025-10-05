export CUDA=0
export CONFIG_PATH="configs/infer.json"
accelerate launch \
  --config_file=configs/accelerate/$CUDA.yaml \
  --mixed_precision="fp16" \
  --main_process_port="12345" \
  infer.py --config_path=$CONFIG_PATH
