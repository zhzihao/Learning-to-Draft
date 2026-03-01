import sys
import os
import wandb
from wandb.integration.sb3 import WandbCallback
# 添加项目根目录到 sys.path，确保可以用绝对路径导入
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import random # Import random for sampling
import time # Make sure time is imported

try:
    from ..model.ea_model import EaModel
    from ..model.kv_cache import initialize_past_key_values
    from ..model.utils import *
except:
    # Assuming 'eagle' is the fallback path you intend
    from eagle.model.ea_model import EaModel
    from eagle.model.kv_cache import initialize_past_key_values
    from eagle.model.utils import *
import json
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
import argparse
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser(description="hello rl")

# ================= 路径相关参数 (Path Arguments) =================
parser.add_argument(
    "--base_model_path",
    type=str,
    default="meta-llama/Llama-3.1-8B-Instruct",
    help="Path or HuggingFace hub name for the base model."
)
parser.add_argument(
    "--ea_model_path",
    type=str,
    default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    help="Path or HuggingFace hub name for the EA model."
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="./eagle/data",
    help="Base directory for the datasets."
)
parser.add_argument(
    "--rl_checkpoint_path",
    type=str,
    default="",
    help="Path to an existing RL checkpoint zip file to resume training."
)
parser.add_argument(
    "--depth_model",
    type=str,
    default="",
    help="Path to the RL depth model zip file (required if --use_dyn_depth is set)."
)
# =================================================================

parser.add_argument("--n_steps", type=int, default=128)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--target_kl", type=float, default=0.02)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate for the optimizer.")
parser.add_argument("--total_timesteps", type=int, default=256, help="Total number of timesteps for training.")
parser.add_argument("--warmup_timesteps", type=int, default=10, help="Number of warmup timesteps for the learning rate schedule.")
parser.add_argument("--eval_freq", type=int, default=10000, help="Frequency (in timesteps) to run evaluation.")
parser.add_argument("--save_path", type=str, default="./")
parser.add_argument("--ent_coef", type=float, default=0)
parser.add_argument("--dataset_train", type=str, default="humaneval")
parser.add_argument('--pi_arch', type=int, nargs='+', default=[512, 256], help="Policy network (pi) architecture. Example: --pi_arch 512 256")
parser.add_argument('--vf_arch', type=int, nargs='+', default=[1024, 512], help="Value network (vf) architecture. Example: --vf_arch 1024 512")
parser.add_argument("--use_dyn_depth", action="store_true")

args=parser.parse_args()

# Dummy Adawm Schedule for completeness
def adawm_schedule(initial_lr: float, warmup_steps: int, total_timesteps: int):
    def func(progress_remaining: float) -> float:
        current_timesteps = total_timesteps * (1 - progress_remaining)
        if current_timesteps < warmup_steps:
            return initial_lr * (current_timesteps / warmup_steps) if warmup_steps > 0 else initial_lr
        else:
            decay_progress = (current_timesteps - warmup_steps) / (total_timesteps - warmup_steps) if (total_timesteps - warmup_steps) > 0 else 1.0
            return initial_lr * (1 - decay_progress)
    return func

class CustomTensorboardCallback(BaseCallback):
    """
    一个自定义的回调函数，用于在每个步骤中直接向 Weights & Biases (wandb) 记录详细信息。
    同时也保留了按频率保存模型的功能。
    """
    def __init__(self, verbose=0, save_freq=args.eval_freq, save_path=args.save_path):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.last_saved_timestep = 0

    def _on_step(self) -> bool:
        if "infos" in self.locals and self.locals["infos"]:
            for info in self.locals["infos"]:
                log_data = {}
                
                if "token_right" in info:
                    log_data["custom/token_right"] = info["token_right"]
                if "t_draft" in info:
                    log_data["custom/t_draft"] = info["t_draft"]
                if "total_token_chosen_action" in info:
                    log_data["custom/total_token_chosen_action"] = info["total_token_chosen_action"]
                if "depth_chosen" in info:
                    log_data["custom/random_depth"] = info["depth_chosen"]
                if "current_seq_len" in info:
                    log_data["custom/current_seq_len"] = info["current_seq_len"]
                if "reward_current_step" in info:
                    log_data["custom/reward_current_step"] = info["reward_current_step"]

                if log_data:
                    wandb.log(log_data, step=self.num_timesteps)

        current_timesteps = self.num_timesteps
        if self.save_freq > 0 and current_timesteps - self.last_saved_timestep >= self.save_freq:
            save_path = os.path.join(self.save_path, f"ppo_speculative_decoder_controller_step_{current_timesteps}")
            self.model.save(save_path)
            if self.verbose > 0:
                print(f"Saving model to {save_path} at timestep {current_timesteps}")
            self.last_saved_timestep = current_timesteps
            
        return True

def load_rl_depth_model(model_path):
    if not model_path:
        raise ValueError("Error: --depth_model path must be provided when --use_dyn_depth is true.")
    print(f"Loading RL depth model from: {model_path}")
    model = PPO.load(model_path, device="cuda")
    policy = model.policy
    policy.to("cuda")
    policy.eval()
    return policy

class SpeculativeDecodingEnv(gym.Env):
    metadata = {'render_modes': [], 'render_fps': 4}

    def __init__(self, model, logits_processor, input_ids_list):
        super(SpeculativeDecodingEnv, self).__init__()

        self.model = model
        self.device = next(model.parameters()).device
        self.input_ids_list = input_ids_list
        if not self.input_ids_list:
            raise ValueError("input_ids_list cannot be empty.")

        self.logits_processor = logits_processor
        self.k_max = 24  # Max number of tokens the agent can choose to verify
        self.max_draft_depth = 12 # Max random depth for generation

        self.action_space = spaces.Discrete(self.k_max)
        self.depth_model = None
        if args.use_dyn_depth:
            self.depth_model = load_rl_depth_model(model_path=args.depth_model)
        
        self.obs_size = 1268
        self.obs_size_depth = 128
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)
        self.observation_space_depth = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size_depth,), dtype=np.float32)
        self.current_input_ids = None
        self.finished_overall_generation = True
        self.current_episode_rewards = []
        self.cu_scores_for_obs = None
        self.ea_layer_top_k = self.model.ea_layer.top_k
        self.random_depth_this_step = 0

    def _get_obs_depth(self):
        obs = np.zeros(self.observation_space_depth.shape, dtype=np.float32)
        offset=0
        draft_position_ids = self.cnet_step/10.0
        position_ids=self.current_input_ids.shape[1]/1000.0

        if self.cu_scores_for_obs!=None:
            last_hidden_state_np = self.cu_scores_for_obs.cpu().detach().numpy().flatten()
            obs[offset : offset + len(last_hidden_state_np)] = last_hidden_state_np
            
        offset+=100
        obs[offset : offset +14] = position_ids
        offset+=14
        obs[offset : offset + 14] = draft_position_ids
        return obs

    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        offset = 0
        hidden_states = self.real_hidden_state_for_obs
        
        draft_position_ids = self.cnet_step / 10.0
        position_ids = self.current_input_ids.shape[1] / 1000.0
        scores = torch.cat(self.scores_list, dim=0).view(-1)
        
        last_hidden_state = hidden_states.squeeze()[-4096:]
        if last_hidden_state.ndim == 0:
            last_hidden_state = last_hidden_state.unsqueeze(0)

        last_hidden_state_np = last_hidden_state.cpu().detach().numpy().flatten()

        scores_np = scores.cpu().detach().numpy().flatten()
        obs[offset:offset + len(scores_np)] = scores_np
        offset += 1210
        
        pos_ids_np = np.full(29, position_ids)
        obs[offset : offset + 29] = pos_ids_np
        offset += 29
        
        draft_pos_ids_np = np.full(29, draft_position_ids)
        obs[offset : offset + 29] = draft_pos_ids_np
        offset+=29
        return obs

    def _get_info(self):
        return {
            "message": "Awaiting verification action.",
            "current_input_ids_length": self.current_input_ids.shape[1] if self.current_input_ids is not None else 0,
            "cnet_step": self.cnet_step
        }

    def _prepare_for_drafting(self, accepted_hidden_state_base, next_token_sampled):
        self.time = 0
        begin_time = time.time()
        
        self.hidden_states_for_topk_ea_layer = accepted_hidden_state_base
        self.input_ids_for_topk_first_pass = torch.cat(
            (self.current_input_ids, next_token_sampled.to(self.current_input_ids.device)), dim=1
        )
        self.current_sample_token_for_topk = self.input_ids_for_topk_first_pass[:, -1]

        self.scores_list = []
        self.parents_list = []
        self.ss_token_list = []
        
        _input_ids_for_ea_layer_first_iter = self.input_ids_for_topk_first_pass[:, 1:]
        self.len_posi_for_topk_loop = _input_ids_for_ea_layer_first_iter.shape[1]

        self.model.ea_layer.reset()

        kv_len = self.model.ea_layer.stable_kv[0][0].shape[2] if hasattr(self.model.ea_layer, "stable_kv") and self.model.ea_layer.stable_kv is not None else 0
        input_ids_for_forward = _input_ids_for_ea_layer_first_iter[:, kv_len:] if kv_len > 0 else _input_ids_for_ea_layer_first_iter
        
        out_hidden, past_key_values_ealayer = self.model.ea_layer(
            self.hidden_states_for_topk_ea_layer,
            input_ids=input_ids_for_forward,
            past_key_values=self.model.ea_layer.stable_kv if kv_len > 0 else None,
            use_cache=True
        )

        self.model.ea_layer.stable_kv = past_key_values_ealayer
        self.current_past_key_values_ealayer = past_key_values_ealayer
        
        last_hidden_ea_layer = out_hidden[:, -1]
        last_headout = self.model.ea_layer.lm_head(self.model.ea_layer.norm(last_hidden_ea_layer))
        last_p = self.model.ea_layer.logsoftmax(last_headout)
        top = torch.topk(last_p, self.ea_layer_top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        
        current_scores_for_topk_loop = topk_p[0]
        self.scores_list.append(current_scores_for_topk_loop[None])
        self.current_scores_for_topk_loop_obs = current_scores_for_topk_loop
        self.parents_list.append(torch.zeros(1, dtype=torch.long, device=current_scores_for_topk_loop.device))

        if self.model.ea_layer.config.vocab_size == self.model.ea_layer.config.draft_vocab_size:
            self.ss_token_list.append(topk_index)
            input_ids_for_next_depth_iter = topk_index
        else:
            mapped_tokens = topk_index + self.model.ea_layer.d2t[topk_index]
            self.ss_token_list.append(mapped_tokens)
            input_ids_for_next_depth_iter = mapped_tokens

        self.current_input_ids_for_topk_depth_iter = input_ids_for_next_depth_iter
        self.current_input_hidden_for_topk_depth_iter = last_hidden_ea_layer[None].repeat(1, self.ea_layer_top_k, 1)
        self.current_tree_mask_for_topk_loop = self.model.ea_layer.tree_mask_init.clone().to(self.device)
        self.current_topk_cs_index_for_loop = torch.arange(self.ea_layer_top_k, device=self.model.ea_layer.embed_tokens.weight.device)
        
        self.real_position_ids_for_obs = torch.tensor([self.input_ids_for_topk_first_pass.shape[1]], device=self.device)
        self.cnet_step = 0
        self.time += time.time() - begin_time

    def _perform_random_depth_expansion(self):
        self.random_depth_this_step = random.randint(1, self.max_draft_depth)
        self.random_depth_this_step=12
        self.entropy_exact=[]
        start_depth_time=time.time()
        for _ in range(self.random_depth_this_step):
            self.model.ea_layer.tree_mask = self.current_tree_mask_for_topk_loop
            current_ea_layer_position_ids = self.len_posi_for_topk_loop + self.model.ea_layer.position_ids.to(self.device)

            out_hidden, past_key_values_ealayer_new = self.model.ea_layer(
                self.current_input_hidden_for_topk_depth_iter,
                input_ids=self.current_input_ids_for_topk_depth_iter,
                past_key_values=self.current_past_key_values_ealayer,
                position_ids=current_ea_layer_position_ids,
                use_cache=True
            )
            self.len_posi_for_topk_loop += 1
            self.current_past_key_values_ealayer = past_key_values_ealayer_new

            bias1 = self.ea_layer_top_k if self.cnet_step > 0 else 0
            bias2 = max(0, self.cnet_step - 1)
            bias = 1 + self.ea_layer_top_k * self.ea_layer_top_k * bias2 + bias1

            parents = self.current_topk_cs_index_for_loop + bias
            self.parents_list.append(parents)

            last_headout = self.model.ea_layer.lm_head(self.model.ea_layer.norm(out_hidden[0]))
            last_p = self.model.ea_layer.logsoftmax(last_headout)
            top = torch.topk(last_p, self.ea_layer_top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + self.current_scores_for_topk_loop_obs[:, None]
            topk_cs = torch.topk(cu_scores.view(-1), self.ea_layer_top_k, dim=-1)
            topk_cs_index_new, topk_cs_p_new = topk_cs.indices, topk_cs.values
            self.cu_scores_for_obs=cu_scores.flatten()
            self.current_scores_for_topk_loop_obs = topk_cs_p_new
            self.current_topk_cs_index_for_loop = topk_cs_index_new

            out_ids = (topk_cs_index_new // self.ea_layer_top_k).to(self.current_tree_mask_for_topk_loop.device)
            self.current_input_hidden_for_topk_depth_iter = out_hidden[:, out_ids]
            next_input_ids_val = topk_index.view(-1)[topk_cs_index_new][None]

            if self.model.ea_layer.config.vocab_size == self.model.ea_layer.config.draft_vocab_size:
                self.ss_token_list.append(topk_index)
                self.current_input_ids_for_topk_depth_iter = next_input_ids_val
            else:
                mapped_tokens = next_input_ids_val + self.model.ea_layer.d2t[next_input_ids_val.squeeze()].unsqueeze(0)
                self.ss_token_list.append(topk_index + self.model.ea_layer.d2t[topk_index.squeeze()])
                self.current_input_ids_for_topk_depth_iter = mapped_tokens
            
            self.scores_list.append(cu_scores)

            if self.current_tree_mask_for_topk_loop.shape[2] > 0 and out_ids.max() < self.current_tree_mask_for_topk_loop.shape[2]:
                self.current_tree_mask_for_topk_loop = torch.cat(
                    (self.current_tree_mask_for_topk_loop[:, :, out_ids],
                     self.model.ea_layer.tree_mask_init.clone().to(self.device)), dim=3
                )
            self.cnet_step += 1
            if self.depth_model!=None and _!=self.random_depth_this_step-1 and _%3==2:
                with torch.inference_mode():
                    depth_tensor=torch.tensor(self._get_obs_depth(),device="cuda")
                    action_depth=self.depth_model._predict(depth_tensor.unsqueeze(0),deterministic=True)[0]
                    if action_depth==0:
                        break
        self.time+=time.time()-start_depth_time

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.finished_overall_generation:
            self.current_episode_rewards = []
            _current_input_ids_tensor = random.choice(self.input_ids_list)
            self.current_input_ids = _current_input_ids_tensor.clone().to(self.device)
            self.input_len = self.current_input_ids.shape[1]
            self.new_token_count = 0

            self.model.ea_layer.reset_kv()
            if hasattr(self.model, "past_key_values") and self.model.past_key_values is not None:
                reset_past_key_values(self.model.past_key_values)
                self.past_key_values = self.model.past_key_values
                self.past_key_values_data = self.model.past_key_values_data
                self.current_length_data = self.model.current_length_data
                self.current_length_data.zero_()
            else:
                (self.past_key_values, self.past_key_values_data, self.current_length_data) = initialize_past_key_values(
                    self.model.base_model, max_length=2048
                )
                self.model.past_key_values = self.past_key_values
                self.model.past_key_values_data = self.past_key_values_data
                self.model.current_length_data = self.current_length_data
            
            reset_tree_mode(self.model)

            with torch.no_grad():
                draft_tokens, retrieve_indices_init, tree_mask_init, tree_position_ids_init, _, _, _ = initialize_tree(
                    self.current_input_ids, self.model, self.past_key_values, self.logits_processor
                )
                self.model.base_model.model.tree_mask = tree_mask_init.to(self.device)
                logits_verify, hidden_state_new_verify, _ = tree_decoding(
                    self.model, draft_tokens.to(self.device), self.past_key_values,
                    tree_position_ids_init.to(self.device), self.current_input_ids, retrieve_indices_init.to(self.device)
                )
                
                padding_init = torch.full((1, 1), -1, dtype=torch.long, device=self.device)
                draft_tokens_padded_init = torch.cat((draft_tokens, padding_init), dim=1)
                candidates_init = draft_tokens_padded_init[0, retrieve_indices_init.to(self.device)]

                best_candidate_idx_init, accept_length_init, sample_p_init = evaluate_posterior(
                    logits_verify, candidates_init, self.logits_processor
                )

                prev_input_len_init = self.current_input_ids.shape[1]
                select_indices = retrieve_indices_init[best_candidate_idx_init, :accept_length_init + 1] + prev_input_len_init
                accepted_tokens_init = candidates_init[best_candidate_idx_init, :accept_length_init + 1]
                self.current_input_ids = torch.cat(
                    (self.current_input_ids, accepted_tokens_init.unsqueeze(0).to(self.current_input_ids.device)), dim=-1
                )

                for pkv_data in self.past_key_values_data:
                    tgt = pkv_data[..., select_indices.to(pkv_data.device), :]
                    dst = pkv_data[..., prev_input_len_init: prev_input_len_init + tgt.shape[-2], :]
                    dst.copy_(tgt, non_blocking=True)

                self.current_length_data.fill_(self.current_input_ids.shape[1])
                retrieve_hidden_state_new = hidden_state_new_verify[:, retrieve_indices_init]
                accepted_hidden_state_base = retrieve_hidden_state_new[:, best_candidate_idx_init, :accept_length_init + 1]
                self.real_hidden_state_for_obs = accepted_hidden_state_base[:, -1, :].unsqueeze(1)
                next_token_sampled = torch.argmax(sample_p_init).unsqueeze(0).unsqueeze(0)
                
            self.new_token_count += accept_length_init + 1
            self._prepare_for_drafting(accepted_hidden_state_base, next_token_sampled)
            self.finished_overall_generation = False
        else:
            self._prepare_for_drafting(self.accepted_hidden_state_base_for_next_topk, self.next_token_sampled_for_next_topk)
        
        self._perform_random_depth_expansion()

        return self._get_obs(), self._get_info()

    def step(self, action):
        start_time_step = time.time()
        
        total_token_val_action = (action + 1)*10
        # --- 1. Finalize Draft Based on Action ---
        _scores_cat_list = torch.cat(self.scores_list, dim=0).view(-1)
        _ss_token_cat_list = torch.cat(self.ss_token_list, dim=0).view(-1)
        _actual_total_tokens = min(_ss_token_cat_list.shape[0], total_token_val_action)
        
        top_scores_indices = torch.topk(_scores_cat_list, _actual_total_tokens, dim=-1).indices
        top_scores_indices_sorted = torch.sort(top_scores_indices).values

        _draft_tokens_flat = _ss_token_cat_list[top_scores_indices_sorted]
        self.finalized_draft_tokens = torch.cat((self.current_sample_token_for_topk.to(self.device), _draft_tokens_flat), dim=0).unsqueeze(0)
        
        _num_final_draft_plus_sample = self.finalized_draft_tokens.shape[1]
        _draft_parents_flat = torch.cat(self.parents_list, dim=0)[top_scores_indices_sorted // self.ea_layer_top_k].long()
        _mask_index = torch.searchsorted(top_scores_indices_sorted, _draft_parents_flat - 1, right=False)
        _mask_index[_draft_parents_flat == 0] = -1
        _mask_index = _mask_index + 1
        _mask_index_list_local = _mask_index.tolist()

        _tree_mask_bool = torch.eye(_num_final_draft_plus_sample, device=self.device).bool()
        _tree_mask_bool[:, 0] = True
        for i in range(_actual_total_tokens):
            parent_idx_in_mask_list = _mask_index_list_local[i]
            _tree_mask_bool[i + 1].add_(_tree_mask_bool[parent_idx_in_mask_list])
            
        self.finalized_tree_mask = _tree_mask_bool.float()[None, None]
        self.finalized_tree_position_ids = torch.sum(_tree_mask_bool.int(), dim=1) - 1

        max_depth = torch.max(self.finalized_tree_position_ids) + 1
        noleaf_index = torch.unique(_mask_index).tolist()
        leaf_num = _actual_total_tokens - (len(noleaf_index)-1)
        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()
        rid = 0
        position_ids_list = self.finalized_tree_position_ids.tolist()
        for i in range(_actual_total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = _mask_index_list_local[cid - 1] if cid > 0 else -1
                rid += 1
        
        self.finalized_retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

        # --- 2. Main Model Verification ---
        self.model.base_model.model.tree_mask = self.finalized_tree_mask.to(self.device)
        logits_verify, hidden_state_new_verify, _ = tree_decoding(
            self.model, self.finalized_draft_tokens.to(self.device), self.past_key_values,
            self.finalized_tree_position_ids.to(self.device), self.current_input_ids, self.finalized_retrieve_indices.to(self.device)
        )

        # --- 3. Evaluate Posterior ---
        padding_verify = torch.full((1, 1), -1, dtype=torch.long, device=self.device)
        draft_tokens_padded_verify = torch.cat((self.finalized_draft_tokens, padding_verify), dim=1)
        _candidates_verify = draft_tokens_padded_verify[0, self.finalized_retrieve_indices.to(self.device)]
        best_candidate_idx, accept_length, sample_p = evaluate_posterior(
            logits_verify, _candidates_verify, self.logits_processor
        )
        self.accept_length = accept_length

        # --- 4. Update Main State ---
        prev_input_len = self.current_input_ids.shape[1]
        accepted_tokens_sequence = _candidates_verify[best_candidate_idx, :self.accept_length + 1]
        self.current_input_ids = torch.cat(
            (self.current_input_ids, accepted_tokens_sequence.unsqueeze(0).to(self.current_input_ids.device)), dim=-1
        )
        num_accepted_this_verification = accepted_tokens_sequence.shape[0]

        select_indices = self.finalized_retrieve_indices[best_candidate_idx, :self.accept_length + 1] + prev_input_len
        for pkv_data in self.past_key_values_data:
            tgt = pkv_data[..., select_indices.to(pkv_data.device), :]
            dst = pkv_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
            dst.copy_(tgt, non_blocking=True)
        self.current_length_data.fill_(self.current_input_ids.shape[1])
        self.new_token_count += num_accepted_this_verification

        # --- 5. Prepare for the NEXT cycle (which will start with reset) ---
        retrieve_hidden_state_new = hidden_state_new_verify[:, self.finalized_retrieve_indices.to(hidden_state_new_verify.device)]
        self.accepted_hidden_state_base_for_next_topk = retrieve_hidden_state_new[:, best_candidate_idx, :self.accept_length + 1]
        self.next_token_sampled_for_next_topk = torch.argmax(sample_p).unsqueeze(0).unsqueeze(0)
        self.real_hidden_state_for_obs = self.accepted_hidden_state_base_for_next_topk[:, -1, :].unsqueeze(1)
        
        self.time += time.time() - start_time_step
        
        # --- 6. Calculate Reward & Termination ---
        reward = (self.accept_length + 1) / (self.time * 100.0 + 1e-6) # Added epsilon for stability
        self.current_episode_rewards.append(reward)
        
        terminated = True
        truncated = False

        stop_token_id = self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>") if self.model.tokenizer else -1
        generated_tokens = self.current_input_ids[0, self.input_len:].tolist()
        if (stop_token_id != -1 and stop_token_id in generated_tokens) or \
           (self.model.tokenizer and self.model.tokenizer.eos_token_id in generated_tokens) or \
           (self.current_input_ids.shape[1] >= 1748) or \
           (self.new_token_count >= 256):
             self.finished_overall_generation = True

        info = self._get_info()
        info.update({
            'token_right': float(self.accept_length + 1),
            't_draft': self.time,
            'total_token_chosen_action': total_token_val_action,
            'depth_chosen': self.random_depth_this_step, 
            'current_seq_len': self.current_input_ids.shape[1],
            'reward_current_step': reward
        })

        obs = self._get_obs()

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    run = wandb.init(
        project="speculative-decoding-rl",  
        config=args,                        
        sync_tensorboard=True,              
        monitor_gym=True,                   
        save_code=True,                     
    )

    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        depth=5,
        top_k=10,
        total_token=60,
        use_eagle3=True,
        use_dyn_len=False,
    ).to("cuda")
    model.eval()
    tokenizer = model.get_tokenizer()

    # Load dataset
    input_ids_list = []
    datasets = [args.dataset_train]
    print(f"Datasets used for RL training: {datasets}")
    for ds1 in datasets:
        dataset_path = os.path.join(args.data_dir, ds1, "question.jsonl")
        with open(dataset_path, "r") as f:
            for line in f:
                data = json.loads(line)
                messages = [
                    {"role": "system",
                    "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
                    {"role": "user", "content": data["turns"][0]}
                ]
                prompt_start = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer.encode(prompt_start, add_special_tokens=False, return_tensors="pt")
                if input_ids.shape[1] <= 1748:
                    input_ids_list.append(input_ids)
    
    # Initialize Environment
    logits_processor = None
    env = SpeculativeDecodingEnv(model, logits_processor, input_ids_list)

    # RL Model Config
    policy_kwargs = dict(net_arch=dict(pi=args.pi_arch, vf=args.vf_arch))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args.warmup_timesteps = 0.01 * args.total_timesteps
    learning_rate_schedule = adawm_schedule(
        args.lr,
        args.warmup_timesteps,
        args.total_timesteps
    )
    
    checkpoint_path = args.rl_checkpoint_path
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        model_rl = PPO.load(
            checkpoint_path,
            env=env,
            learning_rate=learning_rate_schedule,
            tensorboard_log=os.path.join(args.save_path, "ppo_speculative_tensorboard"),
            verbose=1,
            target_kl=args.target_kl,
        )
    else:
        model_rl = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=args.n_steps, 
            batch_size=args.batch_size, 
            n_epochs=args.n_epochs, 
            gamma=args.gamma, 
            tensorboard_log=os.path.join(args.save_path, "ppo_speculative_tensorboard"),
            device=device,
            ent_coef=args.ent_coef, 
            learning_rate=learning_rate_schedule,
        )
        
    custom_tensorboard_callback = CustomTensorboardCallback(save_freq=args.eval_freq)
    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        verbose=2,
    )
    callback_list = CallbackList([custom_tensorboard_callback, wandb_callback])
    
    print("\nStarting RL training with single action (total_tokens)...")
    model_rl.learn(
        total_timesteps=args.total_timesteps, 
        progress_bar=True, 
        callback=callback_list
    )
    print("RL training finished.")
    run.finish()
    model_rl.save(os.path.join(args.save_path, "ppo_speculative_decoder_controller_rebuttal"))
    print("Model saved.")
    env.close()