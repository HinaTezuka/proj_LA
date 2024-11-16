import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron")
sys.path.append("/home/s2410121/proj_LA/activated_neuron/neuron_intervention")
import dill as pickle

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import transformers

from baukit import Trace, TraceDict
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from neuron_intervention_funcs import edit_activation, evaluate_sentence_pair, get_complement, has_overlap

""" load pkl_file(act_freq_dict) """
# en_ja
pkl_file_path = "/home/s2410121/proj_LA/activated_neuron/pickles/act_sum/tatoeba_0_th/shared_neurons_en_ja_tatoeba_0_th.pkl"
with open(pkl_file_path, "rb") as f:
    act_sum_dict = pickle.load(f)
print("unfolded pickle")

"""
list [(layer_idx, neuron_idx, sum), (layer_idx, neuron_idx, sum_of_act_values), ...] <= ちゃんと取れているか確認用
listはact_sumを軸に降順にソート
"""
tuple_list = []
for layer_idx, neurons in act_sum_dict.items():
    for neuron_idx, act_sum in neurons.items():
        tuple_list.append((layer_idx, neuron_idx, act_sum))
tuple_list = sorted(tuple_list, key=lambda x: x[2], reverse=True)

"""
list[(layer_idx, neuron_idx), ...] <= 介入実験用
listはact_sumを軸に降順にソート
"""
layer_neuron_list = []
for layer_idx, neurons in act_sum_dict.items():
    for neuron_idx in neurons.keys():
        layer_neuron_list.append((layer_idx, neuron_idx))
layer_neuron_list = sorted(layer_neuron_list, key=lambda x: act_sum_dict[x[0]][x[1]], reverse=True)

""" 作成したlayer_neuron_listの補集合を作成 """
# 全層数と各層のニューロン数
num_layers = 32
num_neurons_per_layer = 14335
all_layers = range(num_layers)
all_neurons = range(num_neurons_per_layer)
# 補集合の生成
complement_list = get_complement(all_layers, all_neurons, layer_neuron_list)
# 重複がないかテスト
# print(has_overlap(layer_neuron_list, complement_list)) # False
# sys.exit()

# test
# print("Tuple List (act_sum順):", tuple_list[:1000])
# print("Layer-Neuron List (act_sum順):", layer_neuron_list)

# どのくらい介入するか
layer_neuron_list = layer_neuron_list[:2000]
complement_list = complement_list[:2000]

""" neuron intervention (発火値の改竄実験)"""

""" models """
# LLaMA-3
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B"
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    # "nl": "ReBatch/Llama-3-8B-dutch", # du
    # "it": "DeepMount00/Llama-3-8b-Ita", # ita
    # "ko": "beomi/Llama-3-KoEn-8B", # ko
}

# load BLiMP
# BLiMPの評価項目リスト
# configs = get_dataset_config_names("blimp")
configs = configs[:10]
# データを保存するリスト
results = []

def eval_blimp(model_names, layer_neuron_list):
    # 各モデルについてBLiMPのタスクを評価
    for L2, model_name in model_names.items():
        # load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 各評価項目ごとに評価
        for config in configs:
            blimp = load_dataset("blimp", config)
            correct = 0
            total = 0

            for example in blimp["train"]:
                sentence1 = example["sentence_good"]
                sentence2 = example["sentence_bad"]
                score1, score2 = evaluate_sentence_pair(model, tokenizer, layer_neuron_list, sentence1, sentence2)

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
    return results

if __name__ == "__main__":
    result_main = eval_blimp(model_names, layer_neuron_list)
    result_comp = eval_blimp(model_names, complement_list)
    # print(result_main)
    # print(result_comp)

    # データフレームに変換
    df_main = pd.DataFrame(result_main)
    print(df_main)
    df_comp = pd.DataFrame(result_comp)
    print(df_comp)

    # 各モデルごとに正答率の平均を計算
    overall_accuracy_main = df_main.groupby('Model')['Accuracy'].mean().reset_index()
    print(overall_accuracy_main)
    overall_accuracy_comp = df_comp.groupby('Model')['Accuracy'].mean().reset_index()
    print(overall_accuracy_comp)

    # 列名を変更してOVERALLにします
    overall_accuracy.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)

    # CSVに保存
    df_main.to_csv("/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/n_2000/ja/blimp_eval_llama3_en_ja.csv", index=False)
    df_main.to_csv("/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/blimp/n_2000/ja/blimp_eval_llama3_en_ja_COMP.csv", index=False)

    print("評価結果をcsv fileに保存しました。")
