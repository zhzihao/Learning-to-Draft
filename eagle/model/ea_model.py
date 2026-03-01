import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
#from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
from .utils import *
#from .utils_gammatune import *
#from .utils_c2t import *
#from .utils_disco import *
#from .utils_spec_plus import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1
from .cnets_ddd import Model as Modelddd
from .cnets_c2t import Model as Modelc2t
from .cnets_svip import Model as Modelsvip
from .cnets_disco import Model as Modeldisco
from .cnets_gamma import Model as Modelgamma
from .cnets_spec_plus import Model as Modelspecplus
from .configs import EConfig
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
class C2TModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.ffn(x).squeeze(-1)

def load_c2t_model(model_path="/home/v-jiebzhang/GCR1672_EAGLE/eagle/model/c2t_0724.pth"):
    # 初始化模型结构
    model = C2TModel()
    device = torch.device("cuda")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("c2t模型已加载")
    return model

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=12,  hidden_dim=128):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 不加 sigmoid
        )

    def forward(self, x):
        return self.ffn(x).squeeze(-1)

def load_disco_model(model_path="/home/v-jiebzhang/GCR1672_EAGLE/eagle/model/disco_0802.pth"):
    # 初始化模型结构
    model = SimpleClassifier()
    device = torch.device("cuda")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("disco模型已加载")
    return model
class DynamicLengthFFN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256,64], embedding_dim=128, max_draft_len=30):
        super(DynamicLengthFFN, self).__init__()
        
        self.draft_embedding = nn.Embedding(max_draft_len, embedding_dim)
        
        layers = []
        # Adjust the input dimension for the first linear layer
        prev_dim = input_dim + embedding_dim 
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.ffn = nn.Sequential(*layers)
    
    def forward(self, x, draft_len):
        # Ensure draft_len is LongTensor and clamp values to be valid indices [0, max_draft_len-1]
        # Assuming draft_len input is 0-based index corresponding to length 1 onwards
        # Or if draft_len represents actual length (1 to 30), subtract 1
        # Let's assume draft_len is 0-indexed for now. Clamp to [0, 29]
        clamped_draft_len = torch.clamp(draft_len.long(), 0, self.draft_embedding.num_embeddings - 1)
        # Lookup embeddings: shape [batch_size, embedding_dim] or [batch_size*seq_len, embedding_dim]
        # Need to handle potential singleton dimension if draft_len comes in as [N, 1]
        draft_embed = self.draft_embedding(clamped_draft_len.squeeze(-1)) 

        # Concatenate features and draft embeddings
        combined_input = torch.cat([x, draft_embed], dim=-1)
        
        return self.ffn(combined_input)
class TotalTokenPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, max_draft_len=24, embedding_dim=64):
        super(TotalTokenPredictor, self).__init__()
        self.draft_embedding = nn.Embedding(max_draft_len, embedding_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim+embedding_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, features,draft_len):

        draft_embedding = self.draft_embedding(draft_len)
        x = self.relu(self.fc1(features))
        x = self.dropout(x)
        x = self.relu(self.fc2(torch.cat([x, draft_embedding], dim=-1)))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
class SiLUResidualNetD1(nn.Module):
    def __init__(self, input_dim=5120):
        super(SiLUResidualNetD1, self).__init__()
        self.res_block = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.SiLU()
        )
        self.proj = nn.Linear(input_dim, 1024)  # 残差投影层
        self.output_layer = nn.Linear(1024, 1)

    def forward(self, x):
        residual = self.proj(x)  # [batch, 1024]
        out = self.res_block(x)  # [batch, 1024]
        out = out + residual     # [batch, 1024]
        out = self.output_layer(out)
        return out.squeeze(-1)

def load_spec_plus_model(model_path="/home/v-jiebzhang/GCR1672_EAGLE/eagle/model/spec_vicuna.pth"):
    # 初始化模型结构
    model = SiLUResidualNetD1()
    device = torch.device("cuda")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("spec++模型已加载")
    return model
def load_dynamic_length_model(model_path='/home/azureuser/jiebin/GCR1672_EAGLE/predict_models/hid4096_score64_emb128_layer2_train1w/checkpoint_epoch_3.pt'):
    input_dim = 4160
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicLengthFFN(input_dim).to("cuda")
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 将模型设置为评估模式
    model.eval()
    model.to(torch.float32)
    print(f"模型已成功从 {model_path} 加载")
    return model

def load_dynamic_token_model(model_path='/home/v-jiebzhang/EAGLE/models/best_model.pt'):
    input_dim = 2
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TotalTokenPredictor(input_dim=input_dim).to("cuda")
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 将模型设置为评估模式
    model.eval()
    model.to(torch.float32)
    print(f"模型已成功从 {model_path} 加载")
    return model
def load_rl_model(model_path="/home/v-jiebzhang/GCR1672_EAGLE/rl/v15/ppo_speculative_decoder_controller_step_4000000.zip"):
    model = PPO.load(model_path,device="cuda")
    policy=model.policy
    policy.to("cuda")
    policy.eval()
    try:
        # Check for PyTorch 2.0+
        if hasattr(torch, 'compile'):
            if hasattr(policy, 'mlp_extractor') and hasattr(policy.mlp_extractor, 'policy_net'):
                # Using mode="reduce-overhead" is great for small models and fast compilation
                policy.mlp_extractor.policy_net = torch.compile(policy.mlp_extractor.policy_net, mode="reduce-overhead")
                print("  - Successfully compiled 'policy_net'")

            if hasattr(policy, 'mlp_extractor') and hasattr(policy.mlp_extractor, 'value_net'):
                policy.mlp_extractor.value_net = torch.compile(policy.mlp_extractor.value_net, mode="reduce-overhead")
                print("  - Successfully compiled 'value_net'")

            if hasattr(policy, 'action_net'):
                policy.action_net = torch.compile(policy.action_net, mode="reduce-overhead")
                print("  - Successfully compiled 'action_net'")
        else:
            print("  - torch.compile not available (PyTorch < 2.0). Skipping.")
            # Fallback to JIT scripting if you want
            policy.mlp_extractor.policy_net = torch.jit.script(policy.mlp_extractor.policy_net)
    except Exception as e:
        print(f"Warning: Model compilation failed. Error: {e}")
    return policy
def load_rl_token_model(model_path="/home/v-jiebzhang/GCR1672_EAGLE/cloud/ppo_speculative_decoder_controller_step_70000.zip"):
    model = PPO.load(model_path,device="cuda")
    policy=model.policy
    policy_net=policy.mlp_extractor.policy_net
    action_net=policy.action_net
    class PolicyWrapper(nn.Module):

        def __init__(self, policy_net: nn.Module, action_net: nn.Module):


            super().__init__()
            # 直接将传入的模块作为子模块
            self.policy_net = policy_net
            self.action_net = action_net
            
        def forward(self, obs: torch.Tensor) -> torch.Tensor:

            latent_pi = self.policy_net(obs)
            logits = self.action_net(latent_pi)
            return logits
    combined_model = PolicyWrapper(policy_net, action_net).to("cuda")
    #combined_model.to(dtype=torch.float16)
    combined_model.eval()
    return combined_model
def load_rl_depth_model(model_path="/home/v-jiebzhang/GCR1672_EAGLE/cloud/ppo_speculative_decoder_controller_step_500000.zip"):
    model = PPO.load(model_path,device="cpu")
    policy=model.policy
    policy_net=policy.mlp_extractor.policy_net
    action_net=policy.action_net
    class PolicyWrapper(nn.Module):

        def __init__(self, policy_net: nn.Module, action_net: nn.Module):


            super().__init__()
            # 直接将传入的模块作为子模块
            self.policy_net = policy_net
            self.action_net = action_net
            
        def forward(self, obs: torch.Tensor) -> torch.Tensor:

            latent_pi = self.policy_net(obs)
            logits = self.action_net(latent_pi)
            return logits
    combined_model = PolicyWrapper(policy_net, action_net).to("cuda")
    #combined_model.to(dtype=torch.float16)
    combined_model.eval()
    return combined_model
# def load_rl_depth_model(model_path="/home/v-jiebzhang/GCR1672_EAGLE/rl/v17/ppo_speculative_decoder_controller_step_40000.zip"):
#     model = PPO.load(model_path,device="cpu")
#     policy=model.policy
#     policy.to("cpu")
#     policy.eval()

#     return policy
class EaModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
            use_dyn_len=False,
            use_dyn_token=False,
            token_model="",
            depth_model="",
            use_rl=False,
            dyn_depth_type="",
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.use_eagle3 = use_eagle3
        self.accept_length=[]
        self.dyn_tokens=[]
        self.cnet_steps=[]
        self.dtimes=[]
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        if use_eagle3:
            if dyn_depth_type=="ddd":
                print("model ddd")
                self.ea_layer=Modelddd(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
            elif dyn_depth_type=="svip":
                print("model svip")
                self.ea_layer=Modelsvip(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
            elif dyn_depth_type=="gammatune":
                print("model gammatune")
                self.ea_layer=Modelgamma(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
            elif dyn_depth_type=="c2t":
                self.ea_layer=Modelc2t(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
                self.c2t=load_c2t_model()
            elif dyn_depth_type=="disco":
                self.disco=load_disco_model()
                self.ea_layer=Modeldisco(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
            elif dyn_depth_type=="specplus":
                self.spec_plus=load_spec_plus_model()
                self.ea_layer=Modelspecplus(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
            else:
                self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
        else:
            self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)

        low_memory = False
        if use_dyn_len:
            self.dyn_depth_ffn = load_rl_depth_model(model_path=depth_model)
        else:
            self.dyn_depth_ffn = None
        if use_dyn_token:
            self.dyn_token_ffn=load_rl_token_model(model_path=token_model)
        else:
            self.dyn_token_ffn=None
        if use_rl:
            self.policy_net=load_rl_model()
        else:
            self.policy_net=None
        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
            del self.ea_layer.d2t,self.ea_layer.t2d
        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            use_eagle3=True,
            base_model_path=None,
            ea_model_path=None,
            total_token=60,
            depth=7,
            top_k=10,
            threshold=1.0,
            use_dyn_len=False,
            use_dyn_token=False,
            token_model="",
            depth_model="",
            use_rl=False,
            dyn_depth_type="",
            **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == "Qwen3ForCausalLM":
            base_model = KVQwen3ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
            use_dyn_len=use_dyn_len,
            use_dyn_token=use_dyn_token,
            token_model=token_model,
            depth_model=depth_model,
            use_rl=use_rl,
            dyn_depth_type=dyn_depth_type,
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
            pre_len=False,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        self.accept_length=[]
        self.cnet_steps=[]
        self.dyn_tokens=[]
        self.dtimes=[]
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        pre_len_time_total=0
        pre_num_total=0
        for idx in range(max_length):
            #torch.compiler.cudagraph_mark_step_begin()
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token,pre_len_time,pre_num,cnet_step,dyn_token,accept_len,dtime = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )
            self.accept_length.append(int(accept_len))
            self.dyn_tokens.append(int(dyn_token))
            self.cnet_steps.append(int(cnet_step))
            self.dtimes.append(dtime)
            pre_len_time_total+=pre_len_time
            pre_num_total+=pre_num
            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            if pre_len:
                return input_ids, new_token, idx,pre_len_time_total,self.accept_length
            else:
                return input_ids, new_token, idx

    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
