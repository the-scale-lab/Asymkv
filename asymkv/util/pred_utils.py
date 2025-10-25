import torch

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
        if "gemma" in model_name.lower() and (pred_token_idx[0].item() == tokenizer.eos_token_id or tokenizer.decode(pred_token_idx[0]) == "<end_of_turn>"):
            break
    return generated_text

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response
