from asymkv.method.asymkv import enable_asymkv_attention,AsymKVCache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def load_model_and_tokenizer(path, model_name, device,method):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    config= AutoConfig.from_pretrained(path, trust_remote_code=True)
    config._attn_implementation ="eager"
    model = AutoModelForCausalLM.from_pretrained(path,config=config, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    model = model.eval()
    return model, tokenizer 

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len, model_name = "llama"):
    generated_ids = [input_ids.item()]
    pred_token_idx=input_ids
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            spaces_between_special_tokens=False
        ).strip()
        if "llama" in model_name.lower() and (pred_token_idx[0].item() == tokenizer.eos_token_id or pred_token_idx[0].item() == 128009):
            break
        if "qwen" in model_name.lower() and (pred_token_idx[0].item() == tokenizer.eos_token_id or tokenizer.decode(pred_token_idx[0]) == "<|im_end|>"):
            break
        if "mistral" in model_name.lower() and (pred_token_idx[0].item() == tokenizer.eos_token_id or tokenizer.decode(pred_token_idx[0]) == "[/INST]"):
            break
    return generated_text

def asymkv_model_generate(prompt,path,device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    config= AutoConfig.from_pretrained(path, trust_remote_code=True)
    config._attn_implementation ="eager"
    model = AutoModelForCausalLM.from_pretrained(path,config=config, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    model = model.eval()
    enable_asymkv_attention(path,model)
    kv_cache=AsymKVCache(
        start_size=32,
        recent_size=1024,
        k_seq_dim=2,
        v_seq_dim=2,
    )
    past_key_values = None

    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    context_length = input.input_ids.shape[-1]

    input_window = 512
    outputs=None
    hessian_diagnoal = []
    attns_global=[]
    delta_idx=0
    alpha=0
    logits=None
    for idx in range(0, context_length-1,input_window):
        if idx + input_window<context_length:
            input_ids = input.input_ids[:, idx : idx + input_window].to(device)
        elif idx>context_length:
            input_ids = input.input_ids[:, idx-input_window:].to(device)
        elif idx+input_window>=context_length:
            input_ids = input.input_ids[:, idx:].to(device)
        
        if kv_cache is not None and past_key_values is not None and input_ids.shape[1]==input_window:
                num_key_value_groups = model.model.layers[0].self_attn.num_key_value_groups
                for param in model.parameters():
                    param.requires_grad = False
                # past_key_values = tuple(tuple(kv.detach().requires_grad_(True) for kv in layer) for layer in past_key_values)
                key_value_list=[]
                for layer in past_key_values:
                    key = layer[0].detach().requires_grad_(True)
                    value = layer[1].detach().requires_grad_(False) 
                    len = layer[2].detach().requires_grad_(False) 
                    key_value_list.append((key, value,len)) 
                past_key_values = tuple(key_value_list)
                outputs = model(input_ids=input_ids,
                                past_key_values=past_key_values,
                                use_cache=True,
                                output_attentions=True,
                                labels=input_ids)
                loss = outputs.loss
                attns = outputs.attentions

                loss.backward()

                if loss.isnan():
                    hessian_diagnoal_local = None
                    print("loss is Nan for idx", idx)
                else:
                    # 收集 past_key_values 的梯度
                    if not hessian_diagnoal or hessian_diagnoal[0][0].shape[-2]<recent_size+input_window+32:
                        hessian_diagnoal = []
                        for layer in past_key_values:
                            layer_gradients = []
                            for kv in layer[:2]: #kv not len
                                layer_gradients.append(kv.grad)
                            hessian_diagnoal.append(tuple(layer_gradients))
                    else:
                        hessian_diagnoal_local = []
                        for i,layer in enumerate(past_key_values):
                            layer_gradients = []
                            hk_part1 = ((1 / delta_idx)+alpha) * layer[0].grad[:, :, :hessian_diagnoal[i][0].shape[2], :] + \
                                    (((delta_idx - 1) / delta_idx)-alpha) * hessian_diagnoal[i][0]
                            hk_part2 = layer[0].grad[:, :, hessian_diagnoal[i][0].shape[2]:, :]

                            # 拼接前半部分和后半部分
                            hk = torch.cat((hk_part1, hk_part2), dim=2)
                            layer_gradients.append(hk)
                            hv = None
                            layer_gradients.append(hv)
                            hessian_diagnoal_local.append(tuple(layer_gradients))
                        hessian_diagnoal = hessian_diagnoal_local
                    hessian_diagnoal_local = hessian_diagnoal
                torch.cuda.empty_cache()
                
                past_key_values = tuple(tuple(kv.detach().requires_grad_(False) for kv in layer) for layer in past_key_values)
                attns = [attn[:,:,:,:-input_ids.shape[-1]] for attn in attns]
                attns_global=attns
                    
                past_key_values,hessian_diagnoal_local = kv_cache(past_key_values,attns_global,num_key_value_groups, hessian_diagnoal_local,return_Cache=True)
                if hessian_diagnoal_local==[]:
                    hessian_diagnoal=None
                else:
                    hessian_diagnoal=hessian_diagnoal_local
        else:
            with torch.no_grad():
                outputs = model(
                                input_ids,
                                past_key_values=past_key_values,
                                use_cache=True,
                            )
                past_key_values = outputs.past_key_values
                logits= outputs.logits
        delta_idx+=1
    pred_token_idx = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    pred = greedy_generate(
        model, tokenizer, pred_token_idx, past_key_values, max_gen_len=128
    )
    print(pred)

import argparse
def main():
    parser = argparse.ArgumentParser(description="Process data with a specified model and dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model (e.g., '').")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for generation.")
    args = parser.parse_args()
    asymkv_model_generate(prompt=args.prompt,path=args.model_path,device='cuda')

if __name__ == "__main__":
    main()