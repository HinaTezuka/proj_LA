import os
import sys

import torch
import transformers

from baukit import Trace, TraceDict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

"""
上のlayer_neuron_listの補集合を取得
"""
def get_complement(all_layers, all_neurons, original_list):
    """
    特定のニューロンリストから、その補集合を生成する。

    Parameters:
    - all_layers: 全ての層のインデックスのリスト (例: range(num_layers))
    - all_neurons: 各層に含まれるニューロンのインデックスのリスト (例: range(num_neurons_per_layer))
    - original_list: 元のリスト [(layer_idx, neuron_idx), ...]

    Returns:
    - 補集合のリスト [(layer_idx, neuron_idx), ...]
    """
    # 元のリストをセットに変換（高速化のため）
    original_set = set(original_list)

    # 全ての (layer_idx, neuron_idx) を生成
    all_pairs = {(layer_idx, neuron_idx) for layer_idx in all_layers for neuron_idx in all_neurons}

    # 補集合を計算
    complement_set = all_pairs - original_set

    # リストとして返す
    return list(complement_set)

""" かぶっている要素がないか（ちゃんと補集合が取れているか一応確認 """
def has_overlap(list1, list2):
    """
    2つのリストに重複要素があるかを判定する。

    Parameters:
    - list1, list2: リスト

    Returns:
    - True: 重複要素がある場合
    - False: 重複要素がない場合
    """
    # 集合に変換
    set1 = set(list1)
    set2 = set(list2)

    # 積集合が空かどうかをチェック
    return len(set1 & set2) > 0

""" func for editing activation values """
def edit_activation(output, layer, layer_idx_and_neuron_idx):
    """
    edit activation value of neurons(indexed layer_idx and neuron_idx)
    output: activation values
    layer: sth like 'model.layers.{layer_idx}.mlp.act_fn'
    layer_idx_and_neuron_idx: list of tuples like [(layer_idx, neuron_idx), ....]
    """
    for layer_idx, neuron_idx in layer_idx_and_neuron_idx:
        if str(layer_idx) in layer:  # layer名にlayer_idxが含まれているか確認
            if neuron_idx < output.shape[2]:  # ニューロンインデックスが範囲内かチェック
                output[:, :, neuron_idx] *= 0  # 指定されたニューロンの活性化値をゼロに設定
                # print(output[:, :, neuron_idx])
                # print(f"Layer {layer_idx}, Neuron {neuron_idx} activation set to 0")  # 確認用出力
            else:
                print(f"Warning: neuron_idx {neuron_idx} is out of bounds for output with shape {output.shape}")

    return output

def evaluate_sentence_pair_with_edit_activation(model, tokenizer, layer_neuron_list, sentence1, sentence2):
    inputs1 = tokenizer(sentence1, return_tensors="pt").to("cuda")
    inputs2 = tokenizer(sentence2, return_tensors="pt").to("cuda")

    # 指定したニューロンの発火値を改竄した上で対数確率を計算
    trace_layers = [f'model.layers.{layer}.mlp.act_fn' for layer, _ in layer_neuron_list]
    with TraceDict(model, trace_layers, edit_output=lambda output, layer: edit_activation(output, layer, layer_neuron_list)) as tr:
        # input_t = 'こんにちは。今日は'
        # input_ids = tokenizer(input_t, return_tensors="pt")
        # # モデル推論
        # output = model.generate(input_ids["input_ids"])
        # print(tokenizer.decode(output[0], skip_special_tokens=True))
        # sys.exit()

        # logitsを取得
        with torch.no_grad():
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)
        # print(outputs1)
        # sys.exit()

    # 文1の対数確率をトークンごとに取得して平均
    log_probs1 = outputs1.logits.log_softmax(dim=-1)
    score1 = 0.0
    for i in range(inputs1.input_ids.size(1) - 1):
        target_token_id = inputs1.input_ids[0, i + 1]
        score1 += log_probs1[0, i, target_token_id].item()
    score1 /= (inputs1.input_ids.size(1) - 1)  # 平均を取る

    # 文2の対数確率をトークンごとに取得して平均
    log_probs2 = outputs2.logits.log_softmax(dim=-1)
    score2 = 0.0
    for i in range(inputs2.input_ids.size(1) - 1):
        target_token_id = inputs2.input_ids[0, i + 1]
        score2 += log_probs2[0, i, target_token_id].item()
    score2 /= (inputs2.input_ids.size(1) - 1)  # 平均を取る

    return score1, score2
