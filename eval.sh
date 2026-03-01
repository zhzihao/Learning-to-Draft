DATASETS=(alpaca mt_bench qa gsm8k)
Depth_Policy=$1
Size_Policy=$2


for data in "${DATASETS[@]}"; do
    echo "  -> 正在运行数据集: $data"
    
    CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat \
        --bench-name "$data" \
        --depth 8 \
        --temperature 0 \
        --num-choices 1 \
        --total-token 60 \
        --use_dyn_depth \
        --use_dyn_token \
        --token_model ${Size_Policy} \
        --depth_model ${Depth_Policy}
done
