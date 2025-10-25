import math
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from transformers.cache_utils import Cache, StaticCache, SlidingWindowCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers import logging
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv, MistralAttention
from  asymkv.util.cache_utils import apply_rotary_pos_emb_single_withpos, CompressCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2Attention
import types
import logging
logger = logging.getLogger(__name__)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import add_start_docstrings_to_model_forward
from transformers.models.qwen2.modeling_qwen2 import QWEN2_INPUTS_DOCSTRING

cumsum_pos = True
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
def Qwen2Attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    full_len = key_states.shape[-2]
    past_len = 0
    if past_key_value is not None and len(past_key_value.key_cache) > self.layer_idx:
        if cumsum_pos:
            past_len = int(past_key_value.len_cache[self.layer_idx][0,0,:].long().sum())
        else:
            past_len = past_key_value.key_cache[self.layer_idx].shape[-2]
    else:
        past_len = 0
    full_len += past_len

    query_position_ids = torch.arange(past_len, full_len, device=position_ids.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
    if past_key_value is not None and len(past_key_value.key_cache) > self.layer_idx:
        if cumsum_pos:
            key_position_ids = torch.cat((past_key_value.len_cache[self.layer_idx][:,0,:].long().cumsum(dim=-1) - past_key_value.len_cache[self.layer_idx][:,0,0].long().unsqueeze(-1), query_position_ids),dim=1) # bs * seq_len
        else:
            key_position_ids = torch.arange(full_len, device=position_ids.device).unsqueeze(0)
    else:
        key_position_ids = torch.arange(full_len, device=position_ids.device).unsqueeze(0)
    
    cos, sin = self.rotary_emb(value_states, seq_len=full_len)
    len_states = torch.ones((key_states.shape[0], key_states.shape[1], q_len), device=key_states.device, dtype=key_states.dtype)

    if past_key_value is not None:
        key_states, value_states, len_states = past_key_value.update(key_states, value_states, len_states, self.layer_idx)

    query_states = apply_rotary_pos_emb_single_withpos(query_states, cos, sin, position_ids = query_position_ids)
    key_states = apply_rotary_pos_emb_single_withpos(key_states, cos, sin, position_ids = key_position_ids)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    len_states = repeat_kv(len_states.unsqueeze(-1), self.num_key_value_groups).squeeze(-1)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = compute_casual_mask(len_states, q_len, past_len)
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    #attn_weights = torch.exp(attn_weights) * len_states.unsqueeze(-2)
    #attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

    max_score = torch.max(attn_weights, dim=-1, keepdim=True)[0]-30
    attn_weights = attn_weights.to(torch.float32)

    attn_weights = torch.exp((attn_weights-max_score).to(torch.float32))
    attn_weights = attn_weights /(attn_weights* len_states.unsqueeze(-2)).sum(dim=-1, keepdim=True)
    attn_weights = attn_weights.to(query_states.dtype)

    '''attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_weights = attn_weights * len_states.unsqueeze(-2)
    attn_weights = attn_weights.to(query_states.dtype)'''

    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def enable_qwen2_asymkv_attention_recursive(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_qwen2_asymkv_attention_recursive(
                module,
            )

        if isinstance(module, Qwen2Attention):
            model._modules[name].forward = types.MethodType(
                Qwen2Attention_forward, model._modules[name]
            )

def enable_qwen2_asymkv_attention(model):
    model.model.forward = types.MethodType(Qwen2Model_forward, model.model)
    enable_qwen2_asymkv_attention_recursive(model)


@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
def Qwen2Model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    use_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache) and not self.training:
        use_legacy_cache = True
        past_key_values = CompressCache.from_legacy_cache(past_key_values) ###
        logger.warning_once(
            "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
        )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )