# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import copy
import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN

from .utils import Timer
try:
    from .configs import EConfig
    from .utils_c import *
    from .choices import *
except:
    from configs import EConfig
    from utils_c import *
    from choices import *
    from utils import prepare_logits_processor

from concurrent.futures import ThreadPoolExecutor


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            if hasattr(self.config, "rope_theta"):
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings,
                                                       base=self.config.rope_theta)
            else:
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayeremb(nn.Module):
    def __init__(self, config, last=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.last = last
        # self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # if self.index!=0:

        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_emb: torch.Tensor,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)


        # cache_hidden.append(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor



def len_list(x, n):
    return [i for i in x if len(i) <= n]


class Model(nn.Module):
    def __init__(self, config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=8, threshold=1.0):
        super().__init__()
        self.config=config
        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.lm_head=nn.Linear(config.hidden_size,config.draft_vocab_size,bias=False)
        if load_emb and not hasattr(config, "target_hidden_size"):
            from safetensors import safe_open
            import json
            try:
                with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(os.path.join(path, emb_path),
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights = torch.load(os.path.join(path, emb_path))
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor
        self.cpu_executer=ThreadPoolExecutor(max_workers=1)
        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)
        # print("total_tokens",total_tokens)
        # print("depth",depth)
        # print("top_k",top_k)
        # print("threshold",threshold)
        self.hidden_size = config.hidden_size
        self.midlayer = LlamaDecoderLayeremb(config)
        if hasattr(config, "target_hidden_size"):
            self.fc = nn.Linear(config.target_hidden_size * 3, self.hidden_size, bias=False)
        else:
            self.fc = nn.Linear(config.hidden_size * 3, self.hidden_size, bias=False)
        self.norm=LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        d2t=torch.zeros((config.draft_vocab_size),dtype=torch.long)
        t2d=torch.zeros((config.vocab_size),dtype=torch.bool)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)

        for param in self.embed_tokens.parameters():
            param.requires_grad = False
    def init_tree(self):
        self.tree_mask_init = torch.eye(self.top_k, device=self.embed_tokens.weight.device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long)
        self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)

    def reset(self):
        self.tree_mask = None

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min

        return combined_attention_mask

    def forward(
            self,
            hidden_states,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            std=None
    ):
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
            # inputs_embeds = inputs_embeds.detach()

        # if std is not None:
        #     noise = torch.randn(inputs_embeds.size(),device=inputs_embeds.device) * std
        #     inputs_embeds=inputs_embeds+noise

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        #position_ids=position_ids//4
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        # if self.gradient_checkpointing and self.training:
        #    if use_cache:
        #        use_cache = False

        # hidden_states=self.act(self.fc(torch.cat((inputs_embeds,hidden_states),dim=-1)))
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        if hidden_states.shape[-1]!=inputs_embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)
        # hidden_states = self.fc(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None
        past_key_value = past_key_values[0] if past_key_values is not None else None
        layer_outputs = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=True,
        )
        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        hidden_states = layer_outputs[0]


        if use_cache:
            return hidden_states, next_decoder_cache

        return hidden_states

    def reset_kv(self):
        self.stable_kv = None
    def _run_dyn_depth_on_cpu(self,scores_cpu,i,j,dyn_depth_ffn):
        #start_time=time.time()
        obs_buffer_depth = torch.zeros(128, dtype=torch.float32, device="cuda")
        obs_buffer_depth[100:114]=j
        obs_buffer_depth[0:100]=scores_cpu.flatten()
        obs_buffer_depth[114:]=i/10.0
        with torch.inference_mode():
            logits=dyn_depth_ffn(obs_buffer_depth.unsqueeze(0))
            actions=torch.argmax(logits,dim=1)
        # with torch.no_grad():
        #     actions = dyn_depth_ffn._predict(obs_buffer_depth.unsqueeze(0), deterministic=True)
        #end_time=(time.time()-start_time)*1000
        #print("cpu load:",end_time)
        return actions[0]==0
    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor,depth=-1,total_token=-1,obs_tensor=None,policy_net=None,dyn_token_ffn=None,dyn_depth_ffn=None):

        input_ids = input_ids.to(hidden_states.device)
        total_tokens = total_token if total_token!=-1 else self.total_tokens
        depth = depth if depth!=-1 else self.depth
        top_k = self.top_k
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        sample_token = input_ids[:, -1]
        rl_time=0
        scores_list = []
        parents_list = []
        ss_token = []

        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        len_posi = input_ids.shape[1]
        self.reset()

        # with Timer("draft many"):
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                               past_key_values=self.stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        # last_headout = head(last_hidden)
        last_headout = self.lm_head(self.norm(last_hidden))

        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        if self.config.vocab_size==self.config.draft_vocab_size:
            ss_token.append(topk_index)
            input_ids = topk_index
        else:
            ss_token.append(topk_index+self.d2t[topk_index])
            input_ids = topk_index+self.d2t[topk_index]
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        # 4
        
        input_ids_shape_len=input_ids.shape[1]/1000.0
        #start_event.record()
            # end_event.record() # 在GPU流中放置一个结束标记
            # torch.cuda.synchronize()
            # total_gpu_time= start_event.elapsed_time(end_event)
            #print("gpu time:",total_gpu_time)
        cnet_step=0
        cpu_future=None
        entropy_exact=[]
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            # with Timer("draft one"):
            out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                               position_ids=position_ids, use_cache=True)
            len_posi += 1
            # with Timer("sort1"):
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            last_headout = self.lm_head(self.norm(out_hidden[0]))
            last_p = self.logsoftmax(last_headout)
            # p = torch.exp(last_p)
            # ent_tmp=-torch.sum(p * last_p, dim=-1)

            #entropy_exact.append(ent_tmp)
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            out_ids = out_ids.to(tree_mask.device)
            input_hidden = out_hidden[:, out_ids]

            input_ids = topk_index.view(-1)[topk_cs_index][None]

            if self.config.vocab_size == self.config.draft_vocab_size:
                ss_token.append(topk_index)
            else:
                input_ids = input_ids + self.d2t[input_ids]
                ss_token.append(topk_index+self.d2t[topk_index])
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
            rl_btime=time.time()
            # if cpu_future:
            #     should_break = cpu_future.result()
            #     if should_break:
            #     # 这个break是基于上一步(i-1)的计算结果
            #         break
            cnet_step+=1
            if dyn_depth_ffn!=None and i!=depth-1 and i%3==2:
                if self._run_dyn_depth_on_cpu(cu_scores,cnet_step,input_ids_shape_len,dyn_depth_ffn):
                    break
            # if dyn_depth_ffn!=None:
            #     obs_buffer_depth=torch.zeros(128,dtype=torch.float32,device="cpu")
            #     obs_buffer_depth[100:114]=input_ids_shape_len
            #     obs_buffer_depth[0:100]=cu_scores.flatten()
            #     obs_buffer_depth[114:]=i/10.0
            #     with torch.no_grad():
            #         actions = dyn_depth_ffn._predict(obs_buffer_depth.unsqueeze(0), deterministic=True)
            #         print("Num is normal:",actions[0],i)
            #         rl_etime=time.time()
            #         rl_time+=rl_etime-rl_btime
            #         if actions[0]==0:
            #             break
            # if  policy_net!=None:
            #     scores_obs=torch.cat(scores_list,dim=0).view(-1)
            #     obs_tensor[4096:4096 + scores_obs.shape[0]]=scores_obs
            #     obs_tensor[5519:5632].fill_(i/10.0)
                # current_len = 4096 + scores_obs.shape[0]
                # padding_len = 5406 - current_len
                # padding_tensor = torch.zeros(padding_len, dtype=torch.float32, device="cuda")
                # position_tensor = torch.full((113,), position_obs, dtype=torch.float32, device="cuda")
                # i_tensor = torch.full((113,), i / 10.0, dtype=torch.float32, device="cuda")
                # obs_tensor = torch.cat([
                #     last_hidden_obs, 
                #     scores_obs, 
                #     padding_tensor,
                #     position_tensor, 
                #     i_tensor
                # ], dim=0)
                # with torch.no_grad():
                #         #actions,values,log_probs=policy_net.forward(obs_tensor.unsqueeze(0),deterministic=False)
                #     #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                #     actions = policy_net._predict(obs_tensor.unsqueeze(0), deterministic=True)
                #     # print("actions",actions)
                #     # print("values",values)
                #     if actions[0][1]==0:
                #         total_tokens=actions[0][0]+1
                #         break
        rl_btime=time.time()
        if  (policy_net!=None or dyn_token_ffn!=None) and obs_tensor!=None:
            #llama3.1
            scores_obs=torch.cat(scores_list,dim=0).view(-1)
            obs_tensor[0:scores_obs.shape[0]]=scores_obs
            obs_tensor[1239:1268].fill_(cnet_step/10.0)
            # ent_tensor=torch.cat(entropy_exact,dim=0).view(-1).flatten()
            # obs_tensor[1268:1268+len(ent_tensor)]=ent_tensor
            #llama3.1
            # scores_obs=torch.cat(scores_list,dim=0).view(-1)
            # obs_tensor[5120:5120 + scores_obs.shape[0]]=scores_obs
            # obs_tensor[6656-113:6656].fill_(i/10.0)
            with torch.inference_mode():
                logits=dyn_token_ffn(obs_tensor.unsqueeze(0))
                actions=torch.argmax(logits,dim=1)
            #[MOdified]
            total_tokens=(actions[0]+1)*10
            #total_tokens=(actions[0]+1)*10
            #print("total tokens:",total_tokens)
        # if cpu_future and not cpu_future.done():
        #     cpu_future.result()
        end_time=time.time()
        rl_time+=end_time-rl_btime
        #print("predict time:",(end_time-rl_btime)*1000)
        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        total_tokens=min(ss_token_list.shape[0],total_tokens)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        # with Timer("mask"):
        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])


        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        # with Timer("retrieve"):

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                # sort_keys=[len(list)]
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids,rl_time,cnet_step,total_tokens
    @torch.no_grad()
    def topK_genrate_c2t(self, hidden_states, input_ids, head, logits_processor, c2t_ffn="/home/v-jiebzhang/GCR1672_EAGLE/eagle/model/c2t_0724.pth"):

        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]

        scores_list = []
        parents_list = []
        ss_token = []

        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        len_posi = input_ids.shape[1]
        self.reset()

        # with Timer("draft many"):
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                               past_key_values=self.stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        # last_headout = head(last_hidden)
        last_headout = self.lm_head(self.norm(last_hidden))

        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        if self.config.vocab_size==self.config.draft_vocab_size:
            ss_token.append(topk_index)
            input_ids = topk_index
        else:
            ss_token.append(topk_index+self.d2t[topk_index])
            input_ids = topk_index+self.d2t[topk_index]
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)

        # 4
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            # with Timer("draft one"):
            out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                               position_ids=position_ids, use_cache=True)
            len_posi += 1

            # with Timer("sort1"):
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            last_headout = self.lm_head(self.norm(out_hidden[0]))
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]
            
            if i == depth-1:
                depth_tensor = torch.tensor([i + 1], device=cu_scores.device).float() / 10  # 标准化深度
                depth_tensor = depth_tensor.expand_as(cu_scores)

                ffn_input = torch.stack([
                    cu_scores.view(-1),  # 累积概率
                    depth_tensor.view(-1),  # 深度
                ], dim=1)

                with torch.no_grad():
                    keep_scores = c2t_ffn(ffn_input)  # shape: (k*k, 1)
                    keep_flags = (torch.sigmoid(keep_scores.squeeze(-1)) > 0.5)  # bool mask
            
                flat_cu_scores = cu_scores.view(-1)
                filtered_scores = flat_cu_scores[keep_flags]

                if filtered_scores.numel() < top_k:
                    # 若保留不足 top_k，回退为原始 top_k
                    topk_cs = torch.topk(flat_cu_scores, top_k, dim=-1)
                    topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
                    scores = topk_cs_p
                else:
                    # 从过滤后的候选中选 top_k
                    topk_cs = torch.topk(filtered_scores, top_k, dim=-1)
            
                    valid_indices = torch.nonzero(keep_flags, as_tuple=False).squeeze(1)
                    valid_indices = valid_indices.to(cu_scores.device)
                    topk_cs_index = valid_indices[topk_cs.indices]
                    topk_cs_p = topk_cs.values
                    scores = topk_cs_p
            else:
                topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
                topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
                scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]

            input_ids = topk_index.view(-1)[topk_cs_index][None]

            if self.config.vocab_size == self.config.draft_vocab_size:
                ss_token.append(topk_index)
            else:
                input_ids = input_ids + self.d2t[input_ids]
                ss_token.append(topk_index+self.d2t[topk_index])
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)


        scores_list = torch.cat(scores_list, dim=0).view(-1)
        #print(scores_list.shape)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        #print(ss_token_list.shape)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        #print(top_scores.shape)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        # with Timer("mask"):
        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])


        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        # with Timer("retrieve"):

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                # sort_keys=[len(list)]
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids
    
    @torch.no_grad()
    def topK_genrate_dyn_len(self, hidden_states, input_ids, head, logits_processor,dyn_ffn,token_ffn=None):
        # def predict_with_model(model, hidden_state, draft_len, last_p=None):
        #     device = next(model.parameters()).device
        #     mdtype=next(model.parameters()).dtype
        #     epsilon = torch.tensor(1e-12, dtype=mdtype).to(device)
        #     hidden_state = torch.as_tensor(hidden_state.flatten(), dtype=mdtype).to(device)
        #     draft_len = torch.as_tensor(draft_len, dtype=mdtype).view(1).to(device)
        #     prob_dist = torch.clamp(last_p[0], min=epsilon).to(device)
        #     entropy = torch.as_tensor(torch.sum(prob_dist * torch.log(prob_dist), dim=-1).view(1), dtype=mdtype).to(device)
        #     features = torch.cat([hidden_state, draft_len, entropy], dim=0).to(device)
        #     with torch.no_grad():
        #         output = model(features.unsqueeze(0))
        #     probability = output.item()
        #     prediction = 1 if probability > 0.5 else 0
        #     return prediction, probability
        def predict_with_model(model, hidden_state,draft_len,scores):
            device = next(model.parameters()).device
            draft_len +=1
            #draft_lens=torch.full((10, 1), draft_len/30, dtype=torch.float).repeat(1,64).unsqueeze(0).to(device)
            scores=scores.unsqueeze(-1).repeat(1,64).unsqueeze(0)
        # 将特征拼接在一起
            #entropy=entropy.unsqueeze(-1).repeat(1,64).unsqueeze(0)
            features = torch.cat([
                hidden_state,
                scores,
                # entropy,
            ], dim=-1)
            draft_len=torch.full((features.shape[1], 1), int(draft_len), dtype=torch.long, device=features.device)
            with torch.no_grad():
                output = model(features.squeeze(0),draft_len)
                #output=model(features.squeeze(0))
            has_greater_than_half = (output > 0.5).any().item()
            # 找出所有大于0.5的位置
            positions = torch.nonzero(output > 0.5).squeeze().tolist()

            # 如果positions是单个整数（只有一个元素大于0.5），则转换为列表
            if isinstance(positions, int):
                positions = [positions]

            # 返回结果
            result = 1 if has_greater_than_half else 0

            return result, positions
        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]
        entropy_list=[]
        scores_list = []
        parents_list = []
        ss_token = []
        draft_len_list=[]
        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        len_posi = input_ids.shape[1]
        self.reset()

        # with Timer("draft many"):
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                               past_key_values=self.stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        # last_headout = head(last_hidden)
        last_headout = self.lm_head(self.norm(last_hidden))
        # top1k_logits_values, top1k_logits_indices = torch.topk(last_headout, 1000, dim=-1)
        # distribution_top1k = torch.distributions.Categorical(logits=top1k_logits_values)
        # entropy_nats_top1k = distribution_top1k.entropy()
        # entropy_nats_top1k=entropy_nats_top1k.repeat(10)
        # entropy_list.append(entropy_nats_top1k)
        # draft_len_list.append(torch.full((top_k,), 1, dtype=torch.long, device=hidden_states.device))
        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        if self.config.vocab_size==self.config.draft_vocab_size:
            ss_token.append(topk_index)
            input_ids = topk_index
        else:
            ss_token.append(topk_index+self.d2t[topk_index])
            input_ids = topk_index+self.d2t[topk_index]
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        pre_len_time=0
        pre_num=depth
        # 4
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            #with Timer("draft one"):
            out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                               position_ids=position_ids, use_cache=True)
            len_posi += 1

            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            last_headout = self.lm_head(self.norm(out_hidden[0]))
            # top1k_logits_values, top1k_logits_indices = torch.topk(last_headout, 1000, dim=-1)
            # distribution_top1k = torch.distributions.Categorical(logits=top1k_logits_values)
            # entropy_nats_top1k = distribution_top1k.entropy()
            # entropy_nats_top1k=torch.repeat_interleave(entropy_nats_top1k, 10)
            # entropy_list.append(entropy_nats_top1k)
            #draft_len_list.append(torch.full((top_k*top_k,), i+2, dtype=torch.long, device=hidden_states.device))
            last_p = self.logsoftmax(last_headout)
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            #entropy=entropy_nats_top1k[out_ids]
            input_ids = topk_index.view(-1)[topk_cs_index][None]

            if self.config.vocab_size == self.config.draft_vocab_size:
                ss_token.append(topk_index)
            else:
                input_ids = input_ids + self.d2t[input_ids]
                ss_token.append(topk_index+self.d2t[topk_index])
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
            #torch.cuda.synchronize()
            time_start=time.time()
            #with Timer("predict_len"):
            prediction, probability = predict_with_model(dyn_ffn, input_hidden, i,scores)
            #torch.cuda.synchronize()
            time_end=time.time()
            pre_len_time+=time_end-time_start
            if prediction==0:
                pre_num=i+1
                break
            #print(prediction, probability)
            # if prediction==0:
            #     if token_ffn:
            #         scores_token= torch.cat(scores_list, dim=0).view(-1).float()
            #         draft_len_token= torch.cat(draft_len_list, dim=0).view(-1)
            #         entropy_token= torch.cat(entropy_list, dim=0).view(-1).float()
            #         total_tokens=token_ffn(torch.cat([scores_token.unsqueeze(-1),entropy_token.unsqueeze(-1)],dim=-1),draft_len_token)
            #         greater_than_half = torch.sum((total_tokens> 0.5).int()).item()
            #         total_tokens=greater_than_half+1
            #     #print("draft end:",i)
            #     break
        # total_tokens=(i+2)*10
        # print("total_tokens:",total_tokens)
        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        # with Timer("mask"):
        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])


        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        # with Timer("retrieve"):

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long, device=hidden_states.device) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                # sort_keys=[len(list)]
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids,pre_len_time,pre_num

    @torch.no_grad()
    def topK_genrate_with_answers(self, hidden_states, input_ids, head, logits_processor,input_ids_end,dyn_ffn=None,predict_with_models=None):
        input_ids = input_ids.to(hidden_states.device)
        input_ids_end = input_ids_end.to(hidden_states.device)  # 确保input_ids_end在正确的设备上
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]
        train_token=[]
        scores_list = []
        parents_list = []
        ss_token = []
        hit_records = []  # 记录每次迭代中的命中情况
        previous_max_hit_length = 0  # 记录前一次迭代中最长命中长度

        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        len_posi = input_ids.shape[1]
        self.reset()

        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                               past_key_values=self.stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        last_headout = self.lm_head(self.norm(last_hidden))
        top1k_logits_values, top1k_logits_indices = torch.topk(last_headout, 1000, dim=-1)
        distribution_top1k = torch.distributions.Categorical(logits=top1k_logits_values)
        entropy_nats_top1k = distribution_top1k.entropy()
        entropy_nats_top1k=entropy_nats_top1k.repeat(10)
        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        if self.config.vocab_size==self.config.draft_vocab_size:
            ss_token.append(topk_index)
            input_ids = topk_index
        else:
            ss_token.append(topk_index+self.d2t[topk_index])
            input_ids = topk_index+self.d2t[topk_index]
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        draft_len=1
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            tmp={"hidden_state":input_hidden,"input_ids":input_ids,"draft_len":draft_len,"position_ids":position_ids,"flag":torch.zeros(1,input_ids.shape[1],device=input_hidden.device),"scores":scores,"entropy":entropy_nats_top1k}
            out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                               position_ids=position_ids, use_cache=True)
            len_posi += 1

            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)
            last_headout = self.lm_head(self.norm(out_hidden[0]))
            top1k_logits_values, top1k_logits_indices = torch.topk(last_headout, 1000, dim=-1)
            distribution_top1k = torch.distributions.Categorical(logits=top1k_logits_values)
            entropy_nats_top1k = distribution_top1k.entropy()
            last_p = self.logsoftmax(last_headout)
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            # 确保out_ids在与tree_mask相同的设备上
            out_ids = out_ids.to(tree_mask.device)
            
            input_hidden = out_hidden[:, out_ids]
            entropy_nats_top1k=entropy_nats_top1k[out_ids]
            input_ids = topk_index.view(-1)[topk_cs_index][None]

            if self.config.vocab_size == self.config.draft_vocab_size:
                ss_token.append(topk_index)
            else:
                input_ids = input_ids + self.d2t[input_ids]
                ss_token.append(topk_index+self.d2t[topk_index])
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
            
            # 在每次迭代后构建retrieve_indices并检查命中情况
            current_scores_list = torch.cat(scores_list, dim=0).view(-1)
            current_ss_token_list = torch.cat(ss_token, dim=0).view(-1)
            
            # 用当前深度可用的token数量
            current_total_tokens = min(current_ss_token_list.shape[0],total_tokens)
            
            if current_total_tokens > 0:
                # 获取当前最高分的tokens
                top_scores = torch.topk(current_scores_list, current_total_tokens, dim=-1)
                top_scores_index = top_scores.indices
                top_scores_index = torch.sort(top_scores_index).values
                
                current_draft_tokens = current_ss_token_list[top_scores_index]
                current_draft_tokens = torch.cat((sample_token, current_draft_tokens), dim=0)
                
                # 构建当前的树结构
                current_draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
                # 确保设备一致性
                device = current_draft_parents.device
                mask_index = torch.searchsorted(top_scores_index.to(device), current_draft_parents - 1, right=False)
                mask_index[current_draft_parents == 0] = -1
                mask_index = mask_index + 1
                mask_index_list = mask_index.tolist()
                
                current_tree_mask = torch.eye(current_total_tokens + 1, device=device).bool()
                current_tree_mask[:, 0] = True
                for j in range(current_total_tokens):
                    current_tree_mask[j + 1].add_(current_tree_mask[mask_index_list[j]])
                
                current_tree_position_ids = torch.sum(current_tree_mask, dim=1) - 1
                
                # 构建当前的retrieve_indices
                max_depth = torch.max(current_tree_position_ids) + 1
                noleaf_index = torch.unique(mask_index).tolist()
                noleaf_num = len(noleaf_index) - 1
                leaf_num = current_total_tokens - noleaf_num
                
                if leaf_num > 0:
                    retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long, device=device) - 1
                    retrieve_indices = retrieve_indices.tolist()
                    path_scores = []  # 保存每条路径的累计概率分数
                    
                    rid = 0
                    position_ids_list = current_tree_position_ids.tolist()
                    
                    for j in range(current_total_tokens + 1):
                        if j not in noleaf_index:
                            cid = j
                            current_depth = position_ids_list[j]
                            for k in reversed(range(current_depth + 1)):
                                retrieve_indices[rid][k] = cid
                                cid = mask_index_list[cid - 1]
                            
                            # 计算路径对应的累计概率分数
                            valid_indices = [idx for idx in retrieve_indices[rid] if idx >= 0]
                            if valid_indices:
                                path_idx = top_scores_index[valid_indices[-1] - 1] if valid_indices[-1] > 0 else 0
                                path_score = current_scores_list[path_idx].item() if path_idx < len(current_scores_list) else float('-inf')
                                path_scores.append(path_score)
                            else:
                                path_scores.append(float('-inf'))
                            
                            rid += 1
                    
                    # 根据概率分数排序retrieve_indices
                    sorted_indices = sorted(range(len(path_scores)), key=lambda i: path_scores[i], reverse=True)
                    retrieve_indices = [retrieve_indices[i] for i in sorted_indices]
                    path_scores = [path_scores[i] for i in sorted_indices]
                    
                    # 检查retrieve_indices是否命中input_ids_end中的token
                    hit_paths = []
                    hit_tokens = []  # 记录命中路径对应的tokens
                    hit_positions = []  # 记录命中路径在input_ids_end中的位置
                    hit_ranks = []  # 记录命中路径在所有路径中的排名
                    hit_match_lengths = []  # 记录命中的token数量
                    current_max_hit_length = 0  # 当前迭代中最长命中长度
                    hit_flag=0
                    for path_idx, path in enumerate(retrieve_indices):
                        valid_path = [idx for idx in path if idx >= 0]
                        if len(valid_path) > 0:
                            # 确保在同一设备上进行比较
                            path_tokens = current_draft_tokens[valid_path].to(input_ids_end.device)
                            
                            # 只从input_ids_end的开头开始比较，计算最长连续匹配的token数
                            max_possible_match = min(len(path_tokens), input_ids_end.shape[1])
                            match_length = 0
                            
                            for j in range(max_possible_match):
                                if path_tokens[j] == input_ids_end[0, j]:
                                    match_length += 1
                                else:
                                    break
                            
                            # 如果有匹配，记录结果
                            if match_length > 0:
                                current_max_hit_length = max(current_max_hit_length, match_length)
                            if match_length>=draft_len+2:
                                #这里改了一下很可能有问题，需要好好debug验证下
                                match_indices=torch.where(current_draft_parents[path[-1]-1]==parents_list[-1])
                                tmp["flag"][0][match_indices[0].item()]=1
                                break
                    if dyn_ffn:
                        prediction, probability = predict_with_models(dyn_ffn, tmp["hidden_state"], self.embed_tokens(tmp["input_ids"]),tmp["draft_len"])
                        print(probability)
                        print(tmp["flag"])
                    # 打印命中路径的排名信息
                    train_token.append(tmp)
                    draft_len+=1
                    # if hit_ranks:
                    #     # 找出最长匹配的路径
                    #     longest_match_idx = hit_match_lengths.index(max(hit_match_lengths))
                    #     # 获取最长匹配路径的排名
                    #     longest_match_rank = hit_ranks[longest_match_idx]
                        
                    #     print(f"Depth {i}: Longest match length: {hit_match_lengths[longest_match_idx]}, rank: {longest_match_rank} ({longest_match_rank/len(retrieve_indices)*100:.2f}%)")
                    #     print(f"Longest match path: {hit_tokens[longest_match_idx]}")
                        
                    # hit_records.append({
                    #     'depth': i,
                    #     'total_paths': len(retrieve_indices),
                    #     'hit_paths': hit_paths,
                    #     'hit_tokens': hit_tokens,
                    #     'hit_positions': hit_positions,
                    #     'hit_ranks': hit_ranks,
                    #     'hit_match_lengths': hit_match_lengths,
                    #     'max_hit_length': current_max_hit_length
                    # })
                    
                    # if not hit_match_lengths:
                    #     print(f"Depth {i}: No matches found")
                    # else:
                    #     print(f"Depth {i}: Max hit length: {current_max_hit_length}")
                    
                    # 检查是否有改进
                    if current_max_hit_length <= previous_max_hit_length:
                        #print(f"Breaking at depth {i} due to no improvement in hit length")
                        break
                    else:
                        # 更新最长命中长度记录
                        previous_max_hit_length = current_max_hit_length
        return current_max_hit_length, train_token

    @torch.no_grad()
    def topK_genrate_with_tokens_answers(self, hidden_states, input_ids, head, logits_processor,input_ids_end,predict_with_model=None,dyn_ffn=None,token_ffn=None):
        input_ids = input_ids.to(hidden_states.device)
        input_ids_end = input_ids_end.to(hidden_states.device)  # 确保input_ids_end在正确的设备上
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]
        train_token=[]
        scores_list = []
        parents_list = []
        ss_token = []
        hit_records = []  # 记录每次迭代中的命中情况
        previous_max_hit_length = 0  # 记录前一次迭代中最长命中长度

        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        len_posi = input_ids.shape[1]
        self.reset()

        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                               past_key_values=self.stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        last_headout = self.lm_head(self.norm(last_hidden))
        top1k_logits_values, top1k_logits_indices = torch.topk(last_headout, 1000, dim=-1)
        distribution_top1k = torch.distributions.Categorical(logits=top1k_logits_values)
        entropy_nats_top1k = distribution_top1k.entropy()
        entropy_nats_top1k=entropy_nats_top1k.repeat(10)
        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        if self.config.vocab_size==self.config.draft_vocab_size:
            ss_token.append(topk_index)
            input_ids = topk_index
        else:
            ss_token.append(topk_index+self.d2t[topk_index])
            input_ids = topk_index+self.d2t[topk_index]
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        draft_len=1
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            tmp={"hidden_state":input_hidden,"input_ids":input_ids,"draft_len":draft_len,"position_ids":position_ids,"flag":torch.zeros(1,input_ids.shape[1],device=input_hidden.device),"scores":scores,"entropy":entropy_nats_top1k}
            out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                               position_ids=position_ids, use_cache=True)
            len_posi += 1

            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            last_headout = self.lm_head(self.norm(out_hidden[0]))
            top1k_logits_values, top1k_logits_indices = torch.topk(last_headout, 1000, dim=-1)
            distribution_top1k = torch.distributions.Categorical(logits=top1k_logits_values)
            entropy_nats_top1k = distribution_top1k.entropy()
            last_p = self.logsoftmax(last_headout)
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            # 确保out_ids在与tree_mask相同的设备上
            out_ids = out_ids.to(tree_mask.device)
            
            input_hidden = out_hidden[:, out_ids]
            entropy_nats_top1k=entropy_nats_top1k[out_ids]
            input_ids = topk_index.view(-1)[topk_cs_index][None]

            if self.config.vocab_size == self.config.draft_vocab_size:
                ss_token.append(topk_index)
            else:
                input_ids = input_ids + self.d2t[input_ids]
                ss_token.append(topk_index+self.d2t[topk_index])
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
            
            # 在每次迭代后构建retrieve_indices并检查命中情况
            current_scores_list = torch.cat(scores_list, dim=0).view(-1)
            current_ss_token_list = torch.cat(ss_token, dim=0).view(-1)
            
            # 用当前深度可用的token数量
            current_total_tokens = current_ss_token_list.shape[0]
            
            if current_total_tokens > 0:
                # 获取当前最高分的tokens
                top_scores = torch.topk(current_scores_list, current_total_tokens, dim=-1)
                top_scores_index = top_scores.indices
                top_scores_index = torch.sort(top_scores_index).values
                
                current_draft_tokens = current_ss_token_list[top_scores_index]
                current_draft_tokens = torch.cat((sample_token, current_draft_tokens), dim=0)
                
                # 构建当前的树结构
                current_draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
                # 确保设备一致性
                device = current_draft_parents.device
                mask_index = torch.searchsorted(top_scores_index.to(device), current_draft_parents - 1, right=False)
                mask_index[current_draft_parents == 0] = -1
                mask_index = mask_index + 1
                mask_index_list = mask_index.tolist()
                
                current_tree_mask = torch.eye(current_total_tokens + 1, device=device).bool()
                current_tree_mask[:, 0] = True
                for j in range(current_total_tokens):
                    current_tree_mask[j + 1].add_(current_tree_mask[mask_index_list[j]])
                
                current_tree_position_ids = torch.sum(current_tree_mask, dim=1) - 1
                
                # 构建当前的retrieve_indices
                max_depth = torch.max(current_tree_position_ids) + 1
                noleaf_index = torch.unique(mask_index).tolist()
                noleaf_num = len(noleaf_index) - 1
                leaf_num = current_total_tokens - noleaf_num
                
                if leaf_num > 0:
                    retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long, device=device) - 1
                    retrieve_indices = retrieve_indices.tolist()
                    path_scores = []  # 保存每条路径的累计概率分数
                    
                    rid = 0
                    position_ids_list = current_tree_position_ids.tolist()
                    
                    for j in range(current_total_tokens + 1):
                        if j not in noleaf_index:
                            cid = j
                            current_depth = position_ids_list[j]
                            for k in reversed(range(current_depth + 1)):
                                retrieve_indices[rid][k] = cid
                                cid = mask_index_list[cid - 1]
                            
                            # 计算路径对应的累计概率分数
                            valid_indices = [idx for idx in retrieve_indices[rid] if idx >= 0]
                            if valid_indices:
                                path_idx = top_scores_index[valid_indices[-1] - 1] if valid_indices[-1] > 0 else 0
                                path_score = current_scores_list[path_idx].item() if path_idx < len(current_scores_list) else float('-inf')
                                path_scores.append(path_score)
                            else:
                                path_scores.append(float('-inf'))
                            
                            rid += 1
                    
                    # 根据概率分数排序retrieve_indices
                    sorted_indices = sorted(range(len(path_scores)), key=lambda i: path_scores[i], reverse=True)
                    retrieve_indices = [retrieve_indices[i] for i in sorted_indices]
                    path_scores = [path_scores[i] for i in sorted_indices]
                    
                    # 检查retrieve_indices是否命中input_ids_end中的token
                    hit_paths = []
                    hit_tokens = []  # 记录命中路径对应的tokens
                    hit_positions = []  # 记录命中路径在input_ids_end中的位置
                    hit_ranks = []  # 记录命中路径在所有路径中的排名
                    hit_match_lengths = []  # 记录命中的token数量
                    current_max_hit_length = 0  # 当前迭代中最长命中长度
                    hit_flag=0
                    for path_idx, path in enumerate(retrieve_indices):
                        valid_path = [idx for idx in path if idx >= 0]
                        if len(valid_path) > 0:
                            # 确保在同一设备上进行比较
                            path_tokens = current_draft_tokens[valid_path].to(input_ids_end.device)
                            
                            # 只从input_ids_end的开头开始比较，计算最长连续匹配的token数
                            max_possible_match = min(len(path_tokens), input_ids_end.shape[1])
                            match_length = 0
                            
                            for j in range(max_possible_match):
                                if path_tokens[j] == input_ids_end[0, j]:
                                    match_length += 1
                                else:
                                    break
                            
                            # 如果有匹配，记录结果
                            if match_length > 0:
                                current_max_hit_length = max(current_max_hit_length, match_length)
                            if match_length>=draft_len+2:
                                # 获取最后一个节点在current_draft_tokens中的索引
                                node_idx_in_draft = path[-1]
                                rank = -1 # Default rank if not found or it's the initial token

                                if node_idx_in_draft > 0:
                                    # top_scores_index (line 1478) 包含被topk选中的token的原始索引（在current_scores_list中的索引），并按原始索引值排序
                                    # 我们需要找到 path[-1] 对应的原始索引
                                    original_idx = top_scores_index[node_idx_in_draft - 1]

                                    # top_scores.indices (line 1477) 包含按分数排序的原始索引
                                    sorted_by_score_indices = top_scores.indices

                                    # 在按分数排序的索引列表中查找原始索引的位置，即为排名
                                    rank_tensor = torch.where(sorted_by_score_indices == original_idx)[0]
                                    if rank_tensor.numel() > 0:
                                        rank = rank_tensor.item() + 1 # Rank starts from 1
                                match_indices=torch.where(current_draft_parents[path[-1]-1]==parents_list[-1])
                                tmp["flag"][0][match_indices[0].item()]=1
                                tmp["rank"]=rank
                                tmp["scores_list"]=current_scores_list
                                hit_flag=1
                                # input_ffn= self.embed_tokens(input_ids)
                                # prediction, probability = predict_with_model(dyn_ffn, input_hidden, self.embed_tokens(input_ids),i,scores)
                                # #print(prediction, probability)
                                # if prediction==0:
                                #     total_tokens=int((token_ffn(torch.cat((input_hidden.squeeze(0).mean(dim=0),input_ffn.squeeze(0).mean(dim=0),scores)),torch.tensor([i]).to(input_hidden.device))*(100*i+110)).item()+1)
                                #     #print("draft end:",i)
                                #     print("total_tokens:",total_tokens)
                                #     print("rank",rank)
                                #     print("draft len:",draft_len)
                                #     print("---------------next data---------------------")
                                # break
                    if hit_flag==0:
                        tmp["rank"]=current_total_tokens
                        tmp["scores_list"]=current_scores_list
                    # 打印命中路径的排名信息
                    train_token.append(tmp)
                    draft_len+=1
                    # if hit_ranks:
                    #     # 找出最长匹配的路径
                    #     longest_match_idx = hit_match_lengths.index(max(hit_match_lengths))
                    #     # 获取最长匹配路径的排名
                    #     longest_match_rank = hit_ranks[longest_match_idx]
                        
                    #     print(f"Depth {i}: Longest match length: {hit_match_lengths[longest_match_idx]}, rank: {longest_match_rank} ({longest_match_rank/len(retrieve_indices)*100:.2f}%)")
                    #     print(f"Longest match path: {hit_tokens[longest_match_idx]}")
                        
                    # hit_records.append({
                    #     'depth': i,
                    #     'total_paths': len(retrieve_indices),
                    #     'hit_paths': hit_paths,
                    #     'hit_tokens': hit_tokens,
                    #     'hit_positions': hit_positions,
                    #     'hit_ranks': hit_ranks,
                    #     'hit_match_lengths': hit_match_lengths,
                    #     'max_hit_length': current_max_hit_length
                    # })
                    
                    # if not hit_match_lengths:
                    #     print(f"Depth {i}: No matches found")
                    # else:
                    #     print(f"Depth {i}: Max hit length: {current_max_hit_length}")
                    
                    # 检查是否有改进
                    if current_max_hit_length <= previous_max_hit_length:
                        #print(f"Breaking at depth {i} due to no improvement in hit length")
                        break
                    else:
                        # 更新最长命中长度记录
                        previous_max_hit_length = current_max_hit_length
        return current_max_hit_length,train_token
    @torch.no_grad()
    def topK_genrate_all_token_answers(self, hidden_states, input_ids, head, logits_processor,input_ids_end,dyn_ffn=None,predict_with_models=None):
        def find_tensor_in_list(x, y):
            # x是形状为[1]的tensor，y是形状为[1,10]或[10]的tensor
            # 返回一个布尔值（是否存在）和索引（如果存在）
            
            # 确保y是一维的
            if y.dim() > 1:
                y = y.squeeze()
            
            # 找出x在y中的位置
            indices = torch.where(y == x)[0]
            
            # 判断是否存在匹配
            exists = len(indices) > 0
            
            # 如果存在，返回第一个匹配的位置；否则返回-1
            position = indices[0].item() if exists else -1
            
            return exists, position
        input_ids = input_ids.to(hidden_states.device)
        input_ids_end = input_ids_end.to(hidden_states.device)  # 确保input_ids_end在正确的设备上
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]
        train_token=[]
        scores_list = []
        parents_list = []
        draft_len_list=[]
        ss_token = []
        entropy_list=[]
        flag_list=[]
        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        len_posi = input_ids.shape[1]
        self.reset()

        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                               past_key_values=self.stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        last_headout = self.lm_head(self.norm(last_hidden))
        top1k_logits_values, top1k_logits_indices = torch.topk(last_headout, 1000, dim=-1)
        distribution_top1k = torch.distributions.Categorical(logits=top1k_logits_values)
        entropy_nats_top1k = distribution_top1k.entropy()
        entropy_nats_top1k=entropy_nats_top1k.repeat(10)
        entropy_list.append(entropy_nats_top1k)
        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        if self.config.vocab_size==self.config.draft_vocab_size:
            ss_token.append(topk_index)
            input_ids = topk_index
        else:
            ss_token.append(topk_index+self.d2t[topk_index])
            input_ids = topk_index+self.d2t[topk_index]
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        draft_len=1
        draft_len_list.append(torch.full((top_k,), 1, dtype=torch.long, device=hidden_states.device))
        flag=torch.full((top_k,), 0, dtype=torch.long, device=hidden_states.device)
        try:
            a,b=find_tensor_in_list(input_ids_end[0][1],ss_token[-1].squeeze())
        except:
            flag_t=0
            a=0
            print("last input_ids")
        tmp_pos=0
        flag_t=0
        if a>0:
            flag[b]=1
            tmp_pos=b
            flag_t=1
            draft_len+=1
            pos=tmp_pos
        flag_list.append(flag)
        if flag_t==0:
            scores_list = torch.cat(scores_list, dim=0).view(-1)
            ss_token_list = torch.cat(ss_token, dim=0).view(-1)
            entropy_list=torch.cat(entropy_list,dim=0).view(-1)
            draft_len_list=torch.cat(draft_len_list,dim=0).view(-1)
            flag_list=torch.cat(flag_list,dim=0).view(-1)
            return draft_len,scores_list,ss_token_list,entropy_list,draft_len_list,flag_list
        for i in range(min(depth,input_ids_end.shape[1]-2)):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                               position_ids=position_ids, use_cache=True)
            len_posi += 1

            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            last_headout = self.lm_head(self.norm(out_hidden[0]))
            last_p = self.logsoftmax(last_headout)
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values
            top1k_logits_values, top1k_logits_indices = torch.topk(last_headout, 1000, dim=-1)
            distribution_top1k = torch.distributions.Categorical(logits=top1k_logits_values)
            entropy_nats_top1k = distribution_top1k.entropy()
            entropy_nats_top1k=torch.repeat_interleave(entropy_nats_top1k, 10)
            entropy_list.append(entropy_nats_top1k)
            draft_len_list.append(torch.full((top_k*top_k,), i+2, dtype=torch.long, device=hidden_states.device))
            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            # 确保out_ids在与tree_mask相同的设备上
            out_ids = out_ids.to(tree_mask.device)
            
            input_hidden = out_hidden[:, out_ids]

            input_ids = topk_index.view(-1)[topk_cs_index][None]

            if self.config.vocab_size == self.config.draft_vocab_size:
                ss_token.append(topk_index)
            else:
                input_ids = input_ids + self.d2t[input_ids]
                ss_token.append(topk_index+self.d2t[topk_index])
            scores_list.append(cu_scores)
            flag_tmp=torch.full((top_k*top_k,), 0, dtype=torch.long, device=hidden_states.device)
            a,b=find_tensor_in_list(torch.tensor(pos+bias,dtype=torch.long,device=hidden_states.device),parents_list[-1])
            pos=parents_list[-1][b]
            if a:
                flag_t,tmp_pos=find_tensor_in_list(input_ids_end[0][i+2],ss_token[-1][b])
                if flag_t:
                    flag_tmp[10*b+tmp_pos]=1
                    draft_len+=1
                    flag_list.append(flag_tmp)
                else:
                    flag_list.append(flag_tmp)
                    break
            else:
                flag_list.append(flag_tmp)
                break    
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        entropy_list=torch.cat(entropy_list,dim=0).view(-1)
        draft_len_list=torch.cat(draft_len_list,dim=0).view(-1)
        flag_list=torch.cat(flag_list,dim=0).view(-1)
        return draft_len,scores_list,ss_token_list,entropy_list,draft_len_list,flag_list

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    config = EConfig.from_pretrained('config.json')
    model = Model(config, load_emb=False)
    print(model)
