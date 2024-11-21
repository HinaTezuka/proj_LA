import sys
import math

from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from datasets import load_dataset, get_dataset_config_names
import torch
import pandas as pd

# 使用するモデル名のリスト
model_names = [
                # "gpt2", # base model: original gpt2(small) model : en
                "rinna/japanese-gpt2-small", # ja
                # "GroNLP/gpt2-small-dutch", # du
                # "dbmdz/german-gpt2", # ger
                # "GroNLP/gpt2-small-italian", # ita
                # "dbddv01/gpt2-french-small", # fre
                # "skt/kogpt2-base-v2", # ko
                # "datificate/gpt2-small-spanish", # spa
              ]
# device
device = "cuda" if torch.cuda.is_available() else "cpu"
# BLiMPの評価項目リスト
configs = get_dataset_config_names("blimp")


""" calc SLOR scores """
# 文の対数確率を計算する関数
def calculate_log_prob(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    # ログ確率 = -損失 × トークン数
    log_prob = -loss.item() * input_ids.size(1)
    return log_prob

# ユニグラム確率を計算する関数
def calculate_unigram_log_prob(sentence, model, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    log_prob = 0
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        input_ids = torch.tensor([[token_id]]).to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        # トークンごとのログ確率を合計
        log_prob += -loss.item()
    return log_prob

# SLORを計算する関数
def evaluate_sentence_pair(sentence1, sentence2, model, tokenizer):
    log_prob1 = calculate_log_prob(sentence1, model, tokenizer)
    log_prob2 = calculate_log_prob(sentence2, model, tokenizer)
    unigram_log_prob1 = calculate_unigram_log_prob(sentence1, model, tokenizer)
    unigram_log_prob2 = calculate_unigram_log_prob(sentence2, model, tokenizer)
    num_tokens1 = len(tokenizer.tokenize(sentence1))
    num_tokens2 = len(tokenizer.tokenize(sentence2))
    slor1 = (log_prob1 - unigram_log_prob1) / num_tokens1
    slor2 = (log_prob2 - unigram_log_prob2) / num_tokens2

    return slor1, slor2


""" 評価 """
# データを保存するリスト
results = []
# 各モデルについてBLiMPのタスクを評価
for model_name in model_names:
    # モデルとトークナイザーをロード
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 各評価項目ごとに評価
    for config in configs:
        blimp = load_dataset("blimp", config)
        correct = 0
        total = 0

        for example in blimp["train"]:
            sentence1 = example["sentence_good"]
            sentence2 = example["sentence_bad"]
            score1, score2 = evaluate_sentence_pair(sentence1, sentence2, model, tokenizer)

            if score1 > score2:
                correct += 1
            total += 1

        # 精度を計算して結果を保存
        accuracy = correct / total
        results.append({
            "Model": model_name,
            "Task": config,
            "Accuracy": accuracy
        })

# データフレームに変換
df = pd.DataFrame(results)
print(df)

# 各モデルごとに正答率の平均を計算
overall_accuracy = df.groupby('Model')['Accuracy'].mean().reset_index()
print(overall_accuracy)

# 列名を変更してOVERALLにします
overall_accuracy.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)


# CSVに保存
df.to_csv("blimp_evaluation_results_complete2_gpt2_all_final_en_ja.csv", index=False)

print("評価結果をcsv fileに保存しました。")
