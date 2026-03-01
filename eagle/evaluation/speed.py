import json
import numpy as np
from transformers import AutoTokenizer
# 方法名，可自由更换
methods = "-depth-10-rlv17-dyn-totaltokens-step40kdyn-depth-v23-step300w-depthtime-v3p5"
#tokenizer=AutoTokenizer.from_pretrained("/home/v-jiebzhang/hf_models/Qwen3-14B")
# 预训练模型路径
#tokenizer = AutoTokenizer.from_pretrained("/home/v-jiebzhang/hf_models/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("/home/jiebin/hf_models/Meta-Llama-3-8B-Instruct")
#tokenizer=AutoTokenizer.from_pretrained("/home/v-jiebzhang/hf_models/Dpsk-Llama-8B")
# 数据集列表
datasets = ["mt_bench","gsm8k","alpaca" ,"qa"]
#datasets=["qa","gsm8k"]
#datasets=["alpaca"]
for dataset in datasets:
    print(f"=== Dataset: {dataset} ===")

# 拼接文件路径
# jsonl_file = f"/home/v-jiebzhang/GCR1672_EAGLE/{dataset}/llama3p18b-temperature0.0depth-12token-60choices-3specplus.jsonl"
#jsonl_file_base=f"/home/v-jiebzhang/GCR1672_EAGLE/{dataset}/qwen3-temperature1.0baseline.jsonl"
#jsonl_file_base=f"/home/v-jiebzhang/GCR1672_EAGLE/{dataset}/llama3p18b-base-temperature-1.0.jsonl"
#jsonl_file=f"/home/v-jiebzhang/GCR1672_EAGLE/{dataset}/llama3p18b-temperature1.0depth-8token-60choices-3v3500kspecplus.jsonl"
#jsonl_file_base=f"/home/v-jiebzhang/GCR1672_EAGLE/{dataset}/llama3p18b-temperature0.0depth-8token-60choices-1dav.jsonl"
#jsonl_file=f"/home/v-jiebzhang/GCR1672_EAGLE/{dataset}/qwen3-temperature1.0token175depth4choices3.jsonl"
#jsonl_file=f"/home/v-jiebzhang/GCR1672_EAGLE/{dataset}/llama3p18b-temperature-0.0-depth-8token-60ppo_speculative_decoder_controller_step_40000best.jsonl"
#jsonl_file=f"/home/v-jiebzhang/GCR1672_EAGLE/{dataset}/ess-vicuna-70b-fp16-temperature1.0depth-8token-50choices3re.jsonl"
#jsonl_file=f"/home/v-jiebzhang/GCR1672_EAGLE/{dataset}/ess-vicuna-70b-fp16-temperature1.0depth-8token-50choices3retokenrewardthput20kdepthppo_speculative_decoder_controller_step_200000.jsonl"
#jsonl_file_base = f"/home/v-jiebzhang/GCR1672_EAGLE/{dataset}/dpsk-temperature-1.0right.jsonl"
#jsonl_file_base=f"/home/v-jiebzhang/GCR1672_EAGLE/{dataset}/dpsk-temperature0.0depth-8token-60choices3with_acc_lentokenppo_speculative_decoder_controller_step_30000depthppo_speculative_decoder_controller_step_700000.jsonl"
#jsonl_file_base=f"/home/v-jiebzhang/GCR1672_EAGLE/{dataset}/llama3p18b-base-temperature-1.0.jsonl"
#jsonl_file=f"/home/v-jiebzhang/GCR1672_EAGLE/{dataset}/dpsk-temperature1.0depth-8token-60choices3righttokenppo_speculative_decoder_controller_step_30000depthd0f1b400k.jsonl"
#jsonl_file=f"/home/v-jiebzhang/GCR1672_EAGLE/{dataset}/llama3p18b-temperature1.0depth-8token-60choices-31Msvip.jsonl"
    jsonl_file_base=f"/home/jiebin/GRIFFIN/{dataset}/8b/llama38b2_40baseline-temperature-0.0depth-5token-60.jsonl"
    jsonl_file=f"/home/jiebin/GRIFFIN/{dataset}/8b/llama38b2_40debug-temperature-0.0depth-8token-60tokentraintoken70kwithppo_griffin_controller_step_10000tokendepthgriffin_depth_rl_checkpointsppo_griffin_depth_controller_step_1000000depth.jsonl"
    #jsonl_file_base=f"/home/v-jiebzhang/GCR1672_EAGLE/{dataset}/ess-vicuna-70b-fp16-baseline-temperature-1.0choices-3.jsonl"
    # 加载改进方法数据
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    speeds = []
    speeds_all = []
    accept_lens=[]
    for datapoint in data:
        tokens = sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        
        try:
            pre_len = sum(datapoint["choices"][0].get('pre_len_times', []))
        except:
            pre_len = 0
        accept_lens.extend(datapoint["choices"][0].get('pre_num', []))
        speeds.append(tokens / times)
        speeds_all.append(tokens / (times + pre_len))

    # 加载 baseline 数据
    data = []
    with open(jsonl_file_base, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    total_time = 0
    total_token = 0
    speeds0 = []
    for datapoint in data:
        answer = datapoint["choices"][0]['turns']
        tokens = sum(len(tokenizer(i).input_ids) - 1 for i in answer)
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds0.append(tokens / times)
        total_time += times
        total_token += tokens

    # 输出结果
    print(f"=== {dataset} ===")
    #print("ratio:", np.mean(speeds) / np.mean(speeds0))
    print("all_ratio:", np.mean(speeds_all) / np.mean(speeds0))
    print("acc",sum(accept_lens)/len(accept_lens))
    print()
