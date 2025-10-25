from transformers.models.llama.modeling_llama import (
    rotate_half,
)
from transformers import DynamicCache
from typing import Any, Dict, List, Optional, Tuple
import torch



def apply_rotary_pos_emb_single(e, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    e_embed = (e * cos) + (rotate_half(e) * sin)
    return e_embed


def apply_rotary_pos_emb_single_withpos(e, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    e_embed = (e * cos) + (rotate_half(e) * sin)
    return e_embed


class CompressCache(DynamicCache):
    def __init__(self) -> None:
        super().__init__()
        self.len_cache: List[torch.Tensor] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        len_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.len_cache.append(len_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            self.len_cache[layer_idx] = torch.cat([self.len_cache[layer_idx], len_states], dim=-1)

        return self.key_cache[layer_idx], self.value_cache[layer_idx], self.len_cache[layer_idx]
    
    def updatek(
        self,
        key_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        self.key_cache[layer_idx] = key_states


    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states, len_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, len_states, layer_idx)
        return cache

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx], self.len_cache[layer_idx]),)
        return legacy_cache