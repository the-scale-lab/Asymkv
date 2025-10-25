from asymkv.streaming_llm.kv_cache import StartRecentKVCache
import torch
from asymkv.streaming_llm.modify_llama import enable_llama_pos_shift_attention
from  asymkv.util.pred_utils import greedy_generate, post_process
from tqdm import tqdm
import json
import torch.distributed as dist
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    repeat_kv,
    rotate_half
)
import types
import math
import torch.nn as nn

def compress_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This function compresses the hidden states from (batch, num_attention_heads, seqlen, head_dim)
    to (batch, num_key_value_heads, seqlen, head_dim) by taking the mean of every n_rep heads.
    """
    batch, num_attention_heads, slen,head_dim= hidden_states.shape
    num_key_value_heads = num_attention_heads // n_rep
    
    # Reshape the hidden states to (batch, num_key_value_heads, n_rep, slen, head_dim)
    hidden_states = hidden_states.view(batch, num_key_value_heads, n_rep, slen,head_dim)
    
    # Take the mean along the n_rep dimension
    compressed_hidden_states = hidden_states.mean(dim=2)
    
    return compressed_hidden_states

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :].expand(batch, num_key_value_heads, n_rep, slen)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen)
   
class AsymKVCache(StartRecentKVCache):
    def formalize_past_key_values(self, past_key_values):
        ret = []
        for i, (k,v) in enumerate(past_key_values):
            l = torch.ones(past_key_values[i][0].size()[:-1], device=past_key_values[i][0].device, 
                           dtype=past_key_values[i][0].dtype)
            ret.append((k,v,l))
        return tuple(ret)
    
    def __call__(self, past_key_values, attns, num_key_value_groups, hessian_diagnoal = None, return_Cache=False):
        batch_size=512
        if past_key_values is None:
            return None
        if len(past_key_values[0]) == 2:
            past_key_values = self.formalize_past_key_values(past_key_values)

        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values,hessian_diagnoal
        
        new_mid = []
        new_hessian_diagnoal_mid=[]
        for i, (k,v,l) in enumerate(past_key_values):
            mid_k=self.k_slice(k,self.start_size,seq_len)
            mid_v=self.v_slice(v,self.start_size,seq_len)
            mid_l= l[:,:,self.start_size:seq_len]#self.l_slice(l,self.start_size,seq_len)
            
            attns_i = attns[i][:,:,:,self.start_size:]
            attns_i = attns_i.sum(dim=-2)
            repeat_mid_l=repeat_kv(mid_l,int(attns_i.shape[1]/mid_l.shape[1]))
            attns_i=attns_i*repeat_mid_l[:,:,:attns_i.shape[2]]
            attns_i = attns_i[:,:,:-1] + attns_i[:,:,1:]
            # 找到attns_i最后一维的最小的self.recent_size个位置
            weight_i = attns_i.sum(dim=1).squeeze(0)
            
            # l加权
            l_i=mid_l[:,:,:].sum(dim=-2)
            l_i=l_i[:,:-1] + l_i[:,1:]
            l_i=l_i[0]
            gamma=4096/math.log(512)
            # exp压缩
            weight_idx=torch.arange(1, len(weight_i)+1).float()
            sqrt_indices=torch.exp(weight_idx/gamma).to(weight_i.device)
            weight_i=weight_i/sqrt_indices*l_i
            # weight_i[-200:] = 1000
            # if return_Cache:
            mink_indices = weight_i.topk(seq_len - self.cache_size, dim=-1, largest=False).indices


            # 第一步：计算压缩值
            if hessian_diagnoal is None:
                ke = (mid_k[:, :, :-1, :] + mid_k[:, :, 1:, :]) / 2
                ve = (mid_v[:, :, :-1, :] + mid_v[:, :, 1:, :]) / 2
                le = mid_l[:, :, :-1] + mid_l[:, :, 1:]
            else:
                hessian_diagnoal_mid_k = self.k_slice(hessian_diagnoal[i][0].pow(2), self.start_size, seq_len)
                epsilon = 1e-21

                k1 = mid_k[:, :, :-1, :]
                k2 = mid_k[:, :, 1:, :]
                hk1 = hessian_diagnoal_mid_k[:, :, :-1, :]
                hk2 = hessian_diagnoal_mid_k[:, :, 1:, :]
                ke = 1/(hk1+hk2+epsilon) * (k1*hk1 + k2*hk2)
                ve = (mid_v[:, :, :-1, :] + mid_v[:, :, 1:, :])
                le = mid_l[:, :, :-1] + mid_l[:, :, 1:]
                
                hessian_diagnoal_mid_k = self.k_slice(hessian_diagnoal[i][0], self.start_size, seq_len)
                hke=(hessian_diagnoal_mid_k[:, :, :-1, :] + hessian_diagnoal_mid_k[:, :, 1:, :])
            # 创建一个mask来标记需要保留的位置
            mask = torch.ones(mid_k.shape[2], dtype=torch.bool, device=mid_k.device)
            mask[mink_indices + 1] = False
            
            # 对于min_indices位置，使用压缩值
            mid_k[:, :, mink_indices, :] = ke[:, :, mink_indices, :]
            mid_v[:, :, mink_indices, :] = ve[:, :, mink_indices, :]
            # 对于min_indices位置的l，使用l[i]+l[i+1]
            mid_l[:, :, mink_indices] = le[:, :, mink_indices]
            
            # 第二步：删除mink_indices+1位置的元素
            new_mid_k = mid_k[:, :, mask, :]
            new_mid_v = mid_v[:, :, mask, :]
            new_mid_l = mid_l[:, :, mask]
            
            new_mid_l = torch.clip(new_mid_l, max=5)
            if hessian_diagnoal is not None:
                hessian_diagnoal_mid_k[:, :, mink_indices, :]= hke[:, :, mink_indices, :]
                new_hessian_diagnoal_mid_k=hessian_diagnoal_mid_k[:, :, mask, :]
                new_hessian_diagnoal_mid.append((new_hessian_diagnoal_mid_k,None))
            # 将结果添加到new_mid列表中
            new_mid.append((new_mid_k, new_mid_v, new_mid_l))

        return [
            [
                torch.cat([self.k_slice(k, 0, self.start_size),new_k],dim=self.k_seq_dim,),
                torch.cat([self.v_slice(v, 0, self.start_size),new_v],dim=self.v_seq_dim,),
                torch.cat([l[:,:,:self.start_size],new_l],dim=2),
            ]
            for (k,v,l), (new_k, new_v, new_l) in zip(past_key_values, new_mid)
        ],[
            [
                torch.cat([hessian_diagnoal[i][0][:,:,:self.start_size],hessian_diagnoal_mid_k],dim=2),
                None
            ]
            for i, (hessian_diagnoal_mid_k, hessian_diagnoal_mid_v) in enumerate(new_hessian_diagnoal_mid)
        ]
