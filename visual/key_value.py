import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from scipy.stats import pearsonr


def load_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    return model, tokenizer

def truncate_data(data,prompt_format,tokenizer):
    max_length=1000
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
        # if len(tokenized_prompt) > max_length:
        #     prompt = tokenizer.decode(tokenized_prompt[:max_length], skip_special_tokens=True)
        truncated_data.append(prompt)
    return truncated_data

def save_attention_data(attention_data, output_dir, idx):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, f'attn_{idx}.npy'), attention_data)

def cosine_similarity_matrix(values):
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    normalized_values = values / norm
    sim_ij = np.dot(normalized_values, normalized_values.T)
    return sim_ij


def get_attention_points_by_rank(value, sample_size=500):
    """
    从 attn_map 中提取 (rank(attn[i, j]), rank(attn[i, j+1])) 作为散点坐标。
    rank 表示 attn_map 中的排序位置（最小值 rank=0，最大值 rank=seq_len*seq_len-1）。
    
    - attn_map.shape = [seq_len, seq_len]
    - 计算每个元素在整个 attn_map 中的排序位置 (rank)，再将 (attn_map[i, j], attn_map[i, j+1]) 替换为 (rank, rank)。
    - 返回 (x, y) 对的列表，并可选地随机采样以减少数量。
    """
    value=value.cpu().to(torch.float32).numpy()
    sim_ij = cosine_similarity_matrix(value)
    # 生成上三角掩码（True 表示需要掩码的部分）
    mask = np.triu(np.ones_like(sim_ij, dtype=bool))

    # 将上三角部分设为 NaN，以便在热力图中隐藏
    sim_ij_masked = np.where(mask, np.nan, sim_ij)
    n=value.shape[0]
# 生成所有下三角的索引对 (i, j)
    # lower_triangle_indices = [(i, j) for i in range(2, n) for j in range(1,i-1)]
    lower_triangular_values = sim_ij_masked[np.tril_indices_from(sim_ij_masked)]
    # 对值进行排序
    sorted_values = np.sort(lower_triangular_values)
    q=len(sorted_values)
    points=[]
    x_values = []
    y_values = []
    for _ in range(100):
        i=random.randint(5, n-1)
        j=random.randint(1,i-2)
        rank1 = np.searchsorted(sorted_values, sim_ij_masked[i, j]) + 1
        x_values.append(rank1/q)
        rank2 = np.searchsorted(sorted_values, sim_ij_masked[i, j+1]) + 1
        y_values.append(rank2/q)
    return x_values,y_values


import matplotlib.pyplot as plt
import seaborn as sns

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def process_data(data, model, tokenizer, device, output_dir,data_name,name="key"):
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_key_value_heads
    all_correlations = torch.zeros(num_layers, num_heads)
    sample_count = 0
    for idx, item in enumerate(tqdm(data)):
        if data_name=="sharegpt":
            if output_dir=="Llama-2-7b-chat":
                data_string = convert_sharegpt_to_llama2_template(item['conversations'])
            else:
                data_string = data_to_string(item['conversations'], tokenizer)
            inputs = tokenizer(data_string, return_tensors="pt",truncation=True,max_length=1000).to(device)
        else:
            data_string = item
            inputs = tokenizer(data_string, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs,use_cache=True)
        past_key_values = outputs.past_key_values

        # 对每一层的每个head计算相关系数
        for layer_idx in tqdm(range(num_layers)):
            for head_idx in range(past_key_values[0][0].shape[1]):
                delta= 1 if name=="value" else 0
                value = past_key_values[layer_idx][delta][0, head_idx]
                #points = get_attention_points(attn_map, sample_size=2000)
                x_values,y_values = get_attention_points_by_rank(value, sample_size=2000)
                correlation, _ = pearsonr(x_values, y_values)
                all_correlations[layer_idx, head_idx] += correlation
        
        sample_count += 1
    # 计算平均相关系数
    all_correlations /= sample_count
    avg_correlations = torch.tensor(all_correlations).mean(dim=0)

    # 绘制热力图
    plt.figure(figsize=(10, 9))
    torch.save(all_correlations, f'heat/{output_dir}_{data_name}_{name}_avg_correlations.pt')
    heatmap=sns.heatmap(all_correlations, cmap='RdBu', center=0,
                vmin=-0.2, vmax=1,
                xticklabels=range(num_heads),
                yticklabels=range(num_layers),
                cbar_kws={"shrink": 0.8, "ticks": plt.LinearLocator(5), "format": "%.2f"}
                )
     # Plotting the heatmap
    
    # Adjusting the axis labels and font size
    plt.xlabel('Head Id', fontsize=33)
    plt.ylabel('Layer Id', fontsize=33)
    
    # Set custom tick intervals for x and y axes
    x_interval = max(1, num_heads // 4)  # Show approximately 5 ticks on the x-axis
    y_interval = max(1, num_layers // 4)  # Show approximately 5 ticks on the y-axis
    
    plt.xticks(ticks=range(0, num_heads, x_interval), 
               labels=range(0, num_heads, x_interval), 
               fontsize=30, rotation=0)
    plt.yticks(ticks=range(0, num_layers, y_interval), 
               labels=range(0, num_layers, y_interval), 
               fontsize=30, rotation=0)
    # 调整颜色条的字体大小
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=33)  # 设置颜色条刻度标签的字体大小
    cbar.set_ticks([-0.2,0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["-0.2",'0', '0.25', '0.5', '0.75', '1.0'])
    # Setting the title with two lines
    # title = f'{output_dir}\n(Average Spearman corr: {round(avg_correlations.mean().item(), 3)})'
    title = f'Average Spearman corr: {round(avg_correlations.mean().item(), 3)}'
    plt.title(title, fontsize=40, pad=20)
    
    # Saving the figure
    plt.tight_layout()
    # plt.savefig(f'visualize/{output_dir}_heatmap.png')
    plt.savefig(f'visualize/{output_dir}_{data_name}_{name}_heatmap.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    # 计算所有样本的平均相关系数
    # avg_correlations = torch.tensor(all_correlations).mean(dim=0)
    # print(f"平均相关系数: {avg_correlations.mean().item()}")


def data_to_string(data_item, tokenizer):
    # 将多轮对话数据转换成应用了chat模板的字符串
    if isinstance(data_item, list):
        # 假设存在 apply_chat_template 方法
        chat_dialog = []
        for turn in data_item:
            if turn["from"] == "human":
                chat_dialog.append({"role": "user", "content": turn["value"]})
            elif turn["from"] == "gpt":
                chat_dialog.append({"role": "assistant", "content": turn["value"]})
        data_string = tokenizer.apply_chat_template(chat_dialog,tokenize=False)

    else:
        data_string = str(data_item)
    return data_string
def convert_sharegpt_to_llama2_template(conversation):
    llama2_template = ""
    for i, turn in enumerate(conversation):
        if turn["from"] == "human":
            # 用户输入
            user_input = turn["value"]
            # 如果是第一轮对话，直接添加 [INST] 标签
            if i == 0:
                llama2_template += f"<s>[INST] {user_input} [/INST]"
            else:
                # 如果不是第一轮，需要添加 </s> 结束符
                llama2_template += f" </s><s>[INST] {user_input} [/INST]"
        elif turn["from"] == "gpt":
            # 助手回复
            assistant_response = turn["value"]
            llama2_template += f" {assistant_response}"
    # 添加最后的 </s> 结束符
    llama2_template += " </s>"
    return llama2_template

import json
import os
import random
import argparse
def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description="Process data with a specified model and dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model (e.g., 'NousResearch/Llama-2-7b-chat-hf').")
    parser.add_argument("--data_name", type=str, required=True, choices=["multifieldqa_en", "qasper", "sharegpt"], help="Name of the dataset to use.")
    parser.add_argument("--name", type=str, default="key",choices=["key", "value"], help="key or value")
    args = parser.parse_args()
    # output_dir = args.model_path.split("/")[-1]
    # if output_dir=="Llama-2-7b-chat-hf":
    #     output_dir="Llama-2-7b-chat"
    # elif output_dir=="Mistral-7B-Instruct-v0.3":
    #    output_dir="Mistral-7B-Instruct" 
    # print(output_dir)
    output_dir = args.model_path.replace("/","_")
    model_path = args.model_path # 替换为你的模型路径
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    data_name=args.data_name
    # 加载数据集
    if data_name!="sharegpt":
        data = load_dataset("json",data_files=f"data/{data_name}.jsonl")
    else:
        data = load_dataset("shibing624/sharegpt_gpt4")
    dataset=data['train']
    data = [d for d in dataset]  # 筛选前256条数据
    # 截断数据
    data=random.sample(data,20)
    if data_name!="sharegpt":
        dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
        prompt_format = dataset2prompt[data_name]
        truncated_data = truncate_data(data,prompt_format,tokenizer)
    else:
        truncated_data=data
    # 处理数据并保存注意力数据
    process_data(truncated_data, model, tokenizer, device, output_dir,data_name,name=args.name)
    print(output_dir)
if __name__ == '__main__':
    main()