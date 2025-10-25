import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def load_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    return model, tokenizer

def truncate_data(data,prompt_format,tokenizer):
    max_length=3000
    truncated_data = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # prompt=f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt=f"[INST]{prompt}[/INST]"
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        truncated_data.append(prompt)
    return truncated_data

def save_attention_data(attention_data, output_dir, idx,key_or_value="key"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, f'{key_or_value}_{idx}.npy'), attention_data)

def process_data(data, model, tokenizer, device, output_dir,key_or_value="key"):
    i=10
    for idx, item in enumerate(tqdm(data)):
        inputs = tokenizer(item, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True,output_attentions=True)
        print(len(outputs.past_key_values))
        if key_or_value=="key":
            past_key_values = outputs.past_key_values[5][0][0,2,:,:].cpu().to(torch.float32).numpy() 
        else:
            past_key_values = outputs.past_key_values[5][1][0,2,:,:].cpu().to(torch.float32).numpy()
        save_attention_data(past_key_values, output_dir, idx,key_or_value)
        if i >10:
            break
import json
import os
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "/root/agent_workspace/model/llama3.1"  # 替换为你的模型路径
    output_dir = "output_attention_data"  # 输出目录
    key_or_value="value"  # "key" 或 "value"
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_path, device)

    # 加载数据集
    data = load_dataset("json",data_files=f"data/qasper.jsonl")
    dataset=data['train']  # 替换为你的数据集名称
    data = [d for d in dataset]  # 筛选前256条数据
    data = data[:5]
    # 截断数据
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    prompt_format = dataset2prompt['qasper']
    truncated_data = truncate_data(data,prompt_format,tokenizer)

    # 处理数据并保存注意力数据
    process_data(truncated_data, model, tokenizer, device, output_dir,key_or_value)

if __name__ == '__main__':
    main()