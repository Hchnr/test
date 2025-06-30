set -ex

rm -rf log/*
mkdir -p log

for i in $(seq 122); do
    formatted=$(printf "%05d" "$i")
    ckpt_file="model-$formatted-of-00121.safetensors"
    nohup python ./trans.py --file $ckpt_file --model_path /share/project/hcr/models/wenxinyiyan/paddle_internal/ERNIE-45-Turbo/ --out_path /share/project/hcr/models/wenxinyiyan/ernie45T-trans/ > "log/$formatted.log" &
done
