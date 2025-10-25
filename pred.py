import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
from asymkv.method.asymkv import enable_asymkv_attention,AsymKVCache
import torch


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Llama-3.1-8B-Instruct", choices=["gemma-1.1-2b","gemma-1.1-7b","Mistral-7B-Instruct-v0.3", "Llama-3.1-8B-Instruct","Qwen2-7B-Instruct"])
    parser.add_argument('--method', type=str, default="asymkv", choices=["llm","asymkv"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def build_chat(prompt, model_name,tokenizer):
    if "llama2" in model_name:
        prompt=f"[INST]{prompt}[/INST]"
    elif "Llama-3" in model_name:
        prompt=f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "Qwen" in model_name or "Mistral" in model_name:
        chat = [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False,add_generation_prompt=True)
    elif "gemma" in model_name:
        chat = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False,add_generation_prompt=True)
    return prompt

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

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

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(model, tokenizer, rank, world_size, data_all, max_gen, prompt_format, dataset, device, model_name, out_path,method):
    data = data_all[rank::world_size]
    k_seq_dim = v_seq_dim = 2
    recent_size=2048
    if method=="asymkv":
        enable_asymkv_attention(model_name,model)

    for json_obj in tqdm(data, desc=f"Processing dataset {dataset} on rank {rank}"):
        if method=="asymkv":
            kv_cache=AsymKVCache(
                start_size=32,
                recent_size=recent_size,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
            )
        else:
            kv_cache=None
        past_key_values = None
        if dataset in ["lsht", "trec", "triviaqa", "samsum"] or dataset in ["lcc", "repobench-p"]:
            prompt = prompt_format.format(**json_obj)
        else:
            prompt = build_chat(prompt_format.format(**json_obj),model_name,tokenizer)
        
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
            
            if kv_cache is not None and past_key_values is not None:
                if method=="asymkv":
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
                    # print(idx,loss.item())

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
            with torch.no_grad():
                outputs = model(
                                input_ids,
                                past_key_values=past_key_values,
                                use_cache=True,
                            )
                past_key_values = outputs.past_key_values           
            delta_idx+=1
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

        pred = greedy_generate(
            model, tokenizer, pred_token_idx, past_key_values, max_gen_len=max_gen
        )
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
    if world_size>1:
        dist.barrier()  

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend='nccl')

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", \
        #             "hotpotqa", "2wikimqa", "musique", \
        #              "gov_report", "qmsum", "multi_news", \
        #             "trec", "triviaqa", "samsum",  \
        #             "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
        datasets = ["2wikimqa"]
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if local_rank == 0:
        if not os.path.exists("pred"):
            os.makedirs("pred")
        if not os.path.exists("pred_e"):
            os.makedirs("pred_e")

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device,args.method)


    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset("json",data_files=f"data/{dataset}.jsonl")
            data=data['train']
            # data = load_dataset('THUDM/LongBench', f"{dataset}", split='test')
            if not os.path.exists(f"pred/{args.method}/{model_name}"):
                os.makedirs(f"pred/{args.method}/{model_name}")
            out_path = f"pred/{args.method}/{model_name}/{dataset}.jsonl"
        if local_rank == 0 and os.path.exists(out_path):
            os.remove(out_path)
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        get_pred(model, tokenizer, local_rank, world_size, data_all, max_gen, prompt_format, dataset, device, model_name, out_path,args.method)
    if world_size > 1:
        dist.destroy_process_group()