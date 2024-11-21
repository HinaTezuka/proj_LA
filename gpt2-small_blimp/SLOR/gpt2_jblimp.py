import sys
import math

from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from datasets import load_dataset, get_dataset_config_names
import torch
import pandas as pd

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
    if not tokens:
        print(f"No tokens generated for sentence: {sentence}")
        return float('-inf')  # トークンがない場合、非常に低い確率とする

    log_prob = 0
    dummy_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == -1:
            print(f"Unknown token encountered: {token}")
            continue  # 無効なトークンはスキップ

        # 入力をトークン + ダミートークンにする
        input_ids = torch.tensor([[token_id, dummy_id]]).to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        loss = outputs.loss
        if not math.isfinite(loss.item()):
            print(f"Invalid loss encountered for token: {token}")
            continue  # 無効な損失はスキップ

        # トークンごとのログ確率を合計
        log_prob += -loss.item()

    return log_prob

def calculate_log_prob_via_logits(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    with torch.no_grad():
        outputs = model(**inputs)
    # 対数確率を取得
    log_probs = outputs.logits.log_softmax(dim=-1)
    # 次のトークンの対数確率を取得
    next_token_log_probs = log_probs[0, :-1, input_ids[0, 1:]]
    # 文全体の対数確率を計算
    return next_token_log_probs.sum().item()

def calculate_unigram_log_prob_via_logits(sentence, model, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    if not tokens:
        print(f"No tokens generated for sentence: {sentence}")
        return float('-inf')  # トークンがない場合、非常に低い確率とする

    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)]).to(device)
    log_prob = 0

    with torch.no_grad():
        outputs = model(input_ids)
    # 対数確率を取得
    log_probs = outputs.logits.log_softmax(dim=-1)
    # トークンごとの対数確率を計算
    for i, token_id in enumerate(input_ids[0]):
        log_prob += log_probs[0, i, token_id].item()
    return log_prob

# SLORを計算する関数
def evaluate_sentence_pair(sentence1, sentence2, model, tokenizer):
    log_prob1 = calculate_log_prob_via_logits(sentence1, model, tokenizer)
    log_prob2 = calculate_log_prob_via_logits(sentence2, model, tokenizer)
    unigram_log_prob1 = calculate_unigram_log_prob_via_logits(sentence1, model, tokenizer)
    unigram_log_prob2 = calculate_unigram_log_prob_via_logits(sentence2, model, tokenizer)
    num_tokens1 = len(tokenizer.tokenize(sentence1))
    num_tokens2 = len(tokenizer.tokenize(sentence2))
    slor1 = (log_prob1 - unigram_log_prob1) / num_tokens1
    slor2 = (log_prob2 - unigram_log_prob2) / num_tokens2

    return slor1, slor2

model_name = "rinna/japanese-gpt2-small"
jblimp = load_dataset("polm-stability/jblimp")
device = "cuda" if torch.cuda.is_available() else "cpu"

# データを保存するリスト
results = []
predictions = defaultdict(lambda: defaultdict(int))
"""
predictions:
{
"task_name":
        "total": num_of_total_sentence(int)
        "correct": num_of_correct_predictions(int)
}
"""
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

for example in jblimp["train"]:
    task_name = example["phenomenon"]
    sentence1 = example["good_sentence"]
    sentence2 = example["bad_sentence"]
    score1, score2 = evaluate_sentence_pair(sentence1, sentence2, model, tokenizer)
    print(score1, score2)
    if score1 > score2:
        predictions[task_name]["correct"] += 1
    predictions[task_name]["total"] += 1

# 精度を計算して結果を保存
for task_name, scores in predictions.items():
    results.append({
        "Model": model_name,
        "Task": task_name,
        "Accuracy": scores["correct"] / scores["total"]
    })

df = pd.DataFrame(results)
print(df)

# CSVに保存
df.to_csv("/home/s2410121/proj_LA/gpt2-small_blimp/SLOR/csv/gpt2_jblimp.csv", index=False)

print("評価結果をcsv fileに保存しました。")

