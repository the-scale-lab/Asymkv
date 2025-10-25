from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    repeat_kv,
    rotate_half
)
import types
import math
import torch.nn.functional as F
from typing import Optional, Tuple
import torch


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    #x: bs * num_heads * seq_len * dim
    #cos: max_size * dim
    #sin: max_size * dim
    #position_ids: bs * seq_len

    position_ids = position_ids.view(-1).long()

    cos = torch.index_select(cos, 0, position_ids)
    sin = torch.index_select(sin, 0, position_ids)
    cos = cos.reshape(x.shape[0], 1, x.shape[2], x.shape[-1])
    sin = sin.reshape(x.shape[0], 1, x.shape[2], x.shape[-1])

    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

def LlamaRotaryEmbedding_forward(self, x, position_ids):
    #对l的最后一位做累加

    this_len = int(position_ids.view(-1).max() + 1)
    if this_len > self.max_seq_len_cached:
        self._set_cos_sin_cache(seq_len=this_len, device=x.device, dtype=x.dtype)
    
    return (
        self.cos_cached.to(dtype=x.dtype),
        self.sin_cached.to(dtype=x.dtype),
    )


def enable_llama_pos_shift_asymkv_attention_433(model):

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_pos_shift_asymkv_attention_433(
                module,
            )

        if isinstance(module, LlamaAttention):
            model._modules[name].forward = types.MethodType(
                llama_pos_shift_attention_asymkv_forward_433, model._modules[name]
            )

            model._modules[name].rotary_emb.forward = types.MethodType(
                LlamaRotaryEmbedding_forward, model._modules[name].rotary_emb
            )

cumsum_pos = False

def compute_casual_mask(lens, q_len, start_pos):
    # lens: bs * head_num * seq_len
    lens = lens.to(torch.int32)
    real_pos = (lens.cumsum(dim=-1) - lens[:,:,:1]).unsqueeze(-2) # 1, 1, 1, key_len
    # 创建一个位置索引矩阵
    pos_idx = torch.arange(q_len, device=lens.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1) # 1, 1, q_len, 1
    pos_idx += start_pos
    # 根据条件赋值
    causal_mask = torch.where(real_pos <= pos_idx, torch.tensor(0.0, device=lens.device), torch.tensor(-1e20, device=lens.device))
    return causal_mask.to(torch.bfloat16)

def llama_pos_shift_attention_asymkv_forward_433(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    ### Shift Pos: query pos is min(cache_size, idx)
    # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    if past_key_value is not None and cumsum_pos:
        # 计算之前tokens的实际长度总和
        prev_length = int(past_key_value[2][0,0,:].long().sum())
        query_position_ids = torch.arange(int(prev_length), int(prev_length) + q_len, device=position_ids.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
    else:
        query_position_ids = position_ids

    #cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    cos, sin = self.rotary_emb(value_states, query_position_ids)

    query_states = apply_rotary_pos_emb_single(query_states, cos, sin, query_position_ids)
    ###

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    if past_key_value is not None and cumsum_pos:
        key_position_ids = torch.cat((past_key_value[2][:,0,:].long().cumsum(dim=-1) - past_key_value[2][:,0,0].long().unsqueeze(-1), query_position_ids),dim=1) # bs * seq_len
    else:
        key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)

    if past_key_value is None:
        len_states = torch.ones((key_states.shape[0], key_states.shape[1], kv_seq_len), device=key_states.device, dtype=key_states.dtype)
    else:
        len_states = torch.cat([past_key_value[2], 
                                torch.ones((key_states.shape[0], key_states.shape[1], q_len), device=key_states.device, dtype=key_states.dtype)], dim=2)
    past_key_value = (key_states, value_states, len_states)
    len_states = repeat_kv(len_states.unsqueeze(-1), self.num_key_value_groups).squeeze(-1)

    ### Shift Pos: key pos is the pos in cache
    key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
    ###

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

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
        # causal_mask = compute_casual_mask(len_states, q_len, prev_length)
        attn_weights = attn_weights + attention_mask

    #用exp函数和sum函数手动实现softmax
    # attn_weights = torch.exp(attn_weights) * len_states.unsqueeze(-2)
    # attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
    attn_weights = torch.exp(attn_weights.to(torch.float32))
    attn_weights = attn_weights /(attn_weights* len_states.unsqueeze(-2)).sum(dim=-1, keepdim=True)
    attn_weights = attn_weights.to(query_states.dtype)

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value