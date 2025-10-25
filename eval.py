import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
    "arxiv":rouge_score
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--mode', type=str, default="pred_delta")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    if args.e:
        path = f"pred_e/{args.model}/"
    else:
        path = f"{args.mode}/{args.model}/"
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    if args.e:
        out_path = f"pred_e/{args.model}/result.json"
    else:
        out_path = f"{args.mode}/{args.model}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

group1 = ["narrativeqa", "multifieldqa_en", "qasper"]
group2 = ["hotpotqa", "2wikimqa", "musique"]
group3 = ["multi_news", "qmsum", "gov_report"]
group4 = [ "trec", "triviaqa", "samsum"]
group5 = ["passage_count", "passage_retrieval_en"]
group6 = ["lcc", "repobench-p"]

# 计算每个分组的平均值
def calculate_average(group):
    total = sum(scores[key] for key in group if key in scores)
    count = sum(1 for key in group if key in scores)
    return total / count if count > 0 else 0

Single_DocQA = calculate_average(group1)
Multi_DocQA = calculate_average(group2)
Sum = calculate_average(group3)
Few_shot = calculate_average(group4)
Synthetic = calculate_average(group5)
Code = calculate_average(group6)

result = {
    "Single_DocQA": Single_DocQA,
    "Multi_DocQA": Multi_DocQA,
    "Sum": Sum,
    "Few_shot": Few_shot,
    "Synthetic": Synthetic,
    "Code": Code
}

print(f"\n{result['Single_DocQA']:.4f}\t{result['Multi_DocQA']:.4f}\t{result['Sum']:.4f}\t{result['Few_shot']:.4f}\t{result['Synthetic']:.4f}\t{result['Code']:.4f}", file=open(out_path, 'a', encoding="utf-8"))
print(f"\n{result['Single_DocQA']:.4f}\t{result['Multi_DocQA']:.4f}\t{result['Sum']:.4f}\t{result['Few_shot']:.4f}\t{result['Synthetic']:.4f}\t{result['Code']:.4f}")


