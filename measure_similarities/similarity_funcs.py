import numpy as np
import torch

from collections import defaultdict
from torch import nn


""" 類似度などを計算するfuncs """

"""
cosine類似度
"""
def cos_sim(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    tensor1 = tensor1.view(-1)  # 1次元化
    tensor2 = tensor2.view(-1)  # 1次元化
    cos_sim_value = nn.functional.cosine_similarity(tensor1, tensor2, dim=0)
    return cos_sim_value.item()  # スカラー値を返す

"""
差の絶対値
"""
def mean_abs_diff(v1, v2):
  return torch.abs(torch.tensor(v1) - torch.tensor(v2)).mean().item()

"""
align_tensor_sizes(tensor1: torch.Tensor, tensor2: torch.Tensor):
embedding層などのテンソルは言語ごとにサイズが違い、うまく計算できないため、
比較するテンソルのサイズを調整するための関数
"""
def align_tensor_sizes(tensor1: torch.Tensor, tensor2: torch.Tensor):
    # 各次元ごとにサイズを最小値に合わせる
    min_shape = tuple(min(t1_dim, t2_dim) for t1_dim, t2_dim in zip(tensor1.shape, tensor2.shape))
    tensor1 = tensor1[:min_shape[0], :min_shape[1]]  # 2次元テンソルの場合
    tensor2 = tensor2[:min_shape[0], :min_shape[1]]
    return tensor1, tensor2

"""
実際に類似度・差を比較・計算
"""
def compare_between_models(weight_changes: defaultdict(dict), m1_dict: dict, m2_dict: dict) -> defaultdict(dict):
  for layer_name in m1_dict.keys():
    """
    check(if):
    1. layer_nameがもう一方のmodel(m2)のstate_dict.keys()にもあるか（keyの名前がちゃんと一致しているか。してなければスキップ）
    2. 比較対象のlayerのparametersのshapeが一致しているか（していない場合は、align_tensor_sizes()でトリミングし、小さい方のテンソルにサイズを合わせる)
    """
    if layer_name in m2_dict:
      if m1_dict[layer_name].shape != m2_dict[layer_name].shape:
        m1_dict[layer_name], m2_dict[layer_name] = align_tensor_sizes(m1_dict[layer_name], m2_dict[layer_name])
      # 2つの(modelの)parametersの差を計算(絶対値の差の平均)
      weight_mean_abs_diff = mean_abs_diff(m1_dict[layer_name], m2_dict[layer_name])
      # cosine類似度を計算
      cos_sim_value = cos_sim(m1_dict[layer_name], m2_dict[layer_name])
      # 結果をweight_changesに格納
      weight_changes[layer_name]['mean_absolute_difference'] = weight_mean_abs_diff
      weight_changes[layer_name]['cos_sim'] = cos_sim_value
    else:
      continue
  return weight_changes


""" weight_changesのイメージ(weight_changes_gpt2を例に):
{
    'en_ja': {
        'layer_1': {
            'mean_absolute_difference': 0.0123,
            'cosine_similarity': 0.9876,
        },
        'layer_2': {
            'mean_absolute_difference': 0.0456,
            'cosine_similarity': 0.9567,
        },
        # ...他の層も同様に続く...
    },
    'en_du': {
        'layer_1': {
            'mean_absolute_difference': 0.0145,
            'cosine_similarity': 0.9753,
        },
        'layer_2': {
            'mean_absolute_difference': 0.0389,
            'cosine_similarity': 0.9632,
        },
        # ...他の層も同様に続く...
    },
    'en_ger': {
        'layer_1': {
            'mean_absolute_difference': 0.0137,
            'cosine_similarity': 0.9812,
        },
        'layer_2': {
            'mean_absolute_difference': 0.0478,
            'cosine_similarity': 0.9498,
        },
        # ...他の層も同様に続く...
    },
    # ... 他の言語も同様に 'en_ita', 'en_fre', 'en_ko', 'en_spa' が続く...
}
"""
