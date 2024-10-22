import numpy as np
import torch

from collections import defaultdict

# cosine類似度
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 差の絶対値
def mean_abs_diff(v1, v2):
    return torch.abs(torch.tensor(v1) - torch.tensor(v2)).mean().item()

# 上の類似度・差を比較・計算
def compare_between_models(weight_changes: defaultdict(dict), m1_dict: dict, m2_dict: dict) -> defaultdict(dict):
    for layer_name in m1_dict.keys():
        if layer_name in m2_dict: # 同じlayer名のキーがm2_dictにもあるか一応確認（なければスキップ）
           # ２つのモデルの重みの差を計算 (cosine類似度・絶対差の平均)
           cos_sim_value = cos_sim(m1_dict[layer_name], m2_dict[layer_name])
           weight_mean_abs_diff = mean_abs_diff(m1_dict[layer_name], m2_dict[layer_name])
           weight_changes[layer_name]['cosine_similarity'] = cos_sim_value
           weight_changes[layer_name]['mean_absolute_difference'] = weight_mean_abs_diff
        else:
            continue

    return weight_changes


""" weight_changesのイメージ:
{
    'model.layers.0.self_attn.q_proj.weight': {
        'cosine_similarity': 0.98,
        'mean_absolute_difference': 0.0012
    },
    'model.layers.1.self_attn.k_proj.weight': {
        'cosine_similarity': 0.97,
        'mean_absolute_difference': 0.0018
    },
    ...
}
"""

# test
# print(cos_sim([0.1, 2, 3], [2, 1, 1]))
# print(abs_value_diff([0.1, 2, 3], [2, 1, 1]))
