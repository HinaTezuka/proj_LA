import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron")

import numpy as np
import matplotlib.pyplot as plt
import torch
import baukit
from baukit import Trace, TraceDict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModel

# from neuron_detection_funcs import neuron_detection_dict


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
model = AutoModelForCausalLM.from_pretrained(model_names["ja"])
tokenizer = AutoTokenizer.from_pretrained(model_names["ja"])
# Baukit
# def edit_activation(output, layer):
#     # ここでoutputを編集する
#     return output * 0.01  # 活性化値を0にする

# input_t = '太郎さん、こんに'
# with Trace(model, 'model.layers.31.mlp.act_fn', edit_output=edit_activation) as tr:
#     input_ids = tokenizer(input_t, return_tensors="pt")
#     output = model.generate(input_ids["input_ids"])  # モデル推論
#     print("Edited activation values:", tr.output)  # 編集された活性化値
#     print(tokenizer.decode(output[0], skip_special_tokens=True))

import random

# レイヤー番号31は固定
layer_idx = 31

# ニューロンインデックスの1から14000のユニークな数をランダムに選択
neuron_indices = random.sample(range(1, 14001), 1000)

# (レイヤー番号, ニューロンインデックス) のタプルを作成
layer_idx_and_neuron_idx = [(layer_idx, idx) for idx in neuron_indices]

# 結果を確認
print(len(layer_idx_and_neuron_idx))  # 1000
print(layer_idx_and_neuron_idx[:5])  # 最初の5つのタプルを表示
# sys.exit()

def edit_activation(output, layer, layer_idx_and_neuron_idx):
    """
    特定のレイヤーインデックスと複数のニューロンインデックスを指定して活性化値を操作
    :param output: 活性化値
    :param layer: 現在のレイヤー情報
    :param layer_idx_and_neuron_idx: 対象のレイヤーインデックスとニューロンインデックスのリスト
    """
    for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        # 指定されたレイヤーに対して操作を行う
        if str(layer_idx) in layer:
            # ニューロンのインデックスが範囲内かをチェック
            if neuron_idx < output.shape[2]:
                # 指定されたニューロンの活性化値を変更
                # for
                output[:, :, neuron_idx] *= 0
            else:
                print(f"Warning: neuron_idx {neuron_idx} is out of bounds for output with shape {output.shape}")
        else:
            print(f"Warning: layer_idx {layer_idx} is not found in the layer.")

    return output

# モデルの入力を設定
input_t = 'こんにちは。今日は'
input_ids = tokenizer(input_t, return_tensors="pt")
layer_idx = 31

# Traceの開始
with Trace(model, f'model.layers.{layer_idx}.mlp.act_fn', edit_output=lambda output, layer: edit_activation(output, layer, layer_idx_and_neuron_idx)) as tr:
    output = model.generate(input_ids["input_ids"])  # モデル推論
    print("Edited activation values:", tr.output)  # 編集された活性化値
    print(tokenizer.decode(output[0], skip_special_tokens=True))

