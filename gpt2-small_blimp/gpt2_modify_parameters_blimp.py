import sys
sys.path.append('../../proj_LA/measure_similarities')

import torch

from collections import defaultdict
from transformers import AutoModel, GPT2Model

from gpt2_eval_blimp import *
from similarity_funcs import *

# モデルの指定
gpt2_model_original_name = "gpt2" # original gpt2(small) model : en
gpt2_model_japanese_name = "rinna/japanese-gpt2-small" # ja
gpt2_model_dutch_name = "GroNLP/gpt2-small-dutch" # du
gpt2_model_german_name = "dbmdz/german-gpt2" # ger
# gpt2_model_german_name = "ml6team/gpt2-small-german-finetune-oscar" # ger
gpt2_model_italian_name = "GroNLP/gpt2-small-italian" # ita
gpt2_model_french_name = "dbddv01/gpt2-french-small" # fre
gpt2_model_korean_name = "skt/kogpt2-base-v2" # ko
gpt2_model_spanish_name = "datificate/gpt2-small-spanish" # spa

# モデルのロード
gpt2_model_original = GPT2Model.from_pretrained(gpt2_model_original_name) # en
gpt2_model_ja = GPT2Model.from_pretrained(gpt2_model_japanese_name) # ja
gpt2_model_du = GPT2Model.from_pretrained(gpt2_model_dutch_name) # du
gpt2_model_ger = GPT2Model.from_pretrained(gpt2_model_german_name) # ger
gpt2_model_ita = GPT2Model.from_pretrained(gpt2_model_italian_name) # ita
gpt2_model_fre = GPT2Model.from_pretrained(gpt2_model_french_name) # fre
gpt2_model_ko = GPT2Model.from_pretrained(gpt2_model_korean_name) # ko
gpt2_model_spa = GPT2Model.from_pretrained(gpt2_model_spanish_name) # spa

""" それぞれのモデルのparametersを取得 """

# parametersを取得 <- いくつかの辞書ではtransformerという接頭辞がついているため、記法を合わせるために削除(similarity_funcs.pyのdelete_transformer_prefixes_from_state_dict_keysを使用)
state_dict_gpt2_original = gpt2_model_original.state_dict() # original(english)
state_dict_gpt2_ja = delete_transformer_prefixes_from_state_dict_keys(gpt2_model_ja.state_dict()) # japanese
# state_dict_gpt2_du = gpt2_model_du.state_dict() # dutch
# state_dict_gpt2_ger = delete_transformer_prefixes_from_state_dict_keys(gpt2_model_ger.state_dict()) # german
# state_dict_gpt2_ita = gpt2_model_ita.state_dict() # italian
# state_dict_gpt2_fre = gpt2_model_fre.state_dict() # french
# state_dict_gpt2_ko = delete_transformer_prefixes_from_state_dict_keys(gpt2_model_ko.state_dict()) # ko
# state_dict_gpt2_spa = gpt2_model_spa.state_dict() # spanish

# 各モデルのstate_dictのkeyの数、keyの名前が等しいことを確認： 確認済み
# print(state_dict_gpt2_original.keys() == state_dict_gpt2_ja.keys())
# print(state_dict_gpt2_original.keys() == state_dict_gpt2_ja.keys())
# print(state_dict_gpt2_original.keys() == state_dict_gpt2_du.keys())
# print(state_dict_gpt2_original.keys() == state_dict_gpt2_ger.keys())
# print(state_dict_gpt2_original.keys() == state_dict_gpt2_ita.keys())
# print(state_dict_gpt2_original.keys() == state_dict_gpt2_fre.keys())
# print(state_dict_gpt2_original.keys() == state_dict_gpt2_ko.keys())
# print(state_dict_gpt2_original.keys() == state_dict_gpt2_spa.keys())

""" state_dictからoriginl modelと比べて差が大きすぎる特定のparametersを取得(上位kコ)するfunc """
def get_topk_outliers_parameters(m1_dict, m2_dict, topk: int) -> dict:
    """
    絶対値の差が大きすぎるlayerのlayer_nameとparameters(torch.Tensor)をタプルペアとして保存するようのdict(ただしtopk個:更新対象のみ保存:更新対象は L2 modelのparameters)
    key: 絶対値の差(float)
    value: (layer_name: str, m1_dict[layer_name]: torch.Tensor) <- tuple
    """
    abs_diff_descending_dict = {}

    for layer_name in m1_dict.keys():
        # shapeが違う層があったら小さい方のTensorにそろえる
        if m1_dict[layer_name].shape != m2_dict[layer_name].shape:
            m1_dict[layer_name], m2_dict[layer_name] = align_tensor_sizes(m1_dict[layer_name], m2_dict[layer_name])

        # 2つの(modelの)parametersの差を計算(絶対値の差の平均)
        weight_mean_abs_diff = mean_abs_diff(m1_dict[layer_name], m2_dict[layer_name])

        # 辞書に追加
        if len(abs_diff_descending_dict) < topk:
            abs_diff_descending_dict[weight_mean_abs_diff] = (layer_name, m2_dict[layer_name])
        else:
            min_key = min(abs_diff_descending_dict.keys())
            if weight_mean_abs_diff > min_key:
                # 最小要素を削除して新しい要素を追加
                abs_diff_descending_dict.pop(min_key)
                abs_diff_descending_dict[weight_mean_abs_diff] = (layer_name, m2_dict[layer_name])

    # 絶対値の差で降順にソート
    return dict(sorted(abs_diff_descending_dict.items(), reverse=True))

""" topk個の差がある、L2_modelのparametersを緩和するfunc """
def relax_abs_diff_of_L2_model(original_model, L2_model, topk_abs_diff_dict):
    """
    元の差を1/2程度に緩和した新しいL2モデルを返す
    """
    # 新しいモデルを作成
    modified_L2_model = L2_model.copy()  # L2_modelのコピーを作成

    # アウトライヤーのパラメータを緩和
    for abs_diff, (layer_name, l2_param) in topk_abs_diff_dict.items():
        original_param = original_model[layer_name]  # 元のモデルのパラメータ
        # 差を1/2程度に緩和する
        adjusted_param = original_param + ((l2_param - original_param) / 2)
        # modified L2モデルのパラメータを更新
        modified_L2_model[layer_name] = adjusted_param

    return modified_L2_model


""" ちゃんと緩和されているかテスト """
# topk = 20
# topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_ja, topk)
# modified_L2_model = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_ja, topk_outliers)
# print(topk_outliers.keys())
# print('\n')
# print('---------------------------------------------------------------')
# print('\n')
# topk_outliers_modified = get_topk_outliers_parameters(state_dict_gpt2_original, modified_L2_model, topk)
# print(topk_outliers_modified.keys())

""" それぞれのL2モデルのmodified versionを作成 """
topk = 20
# japanese
topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_ja, topk)
modified_ja_model_dict = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_ja, topk_outliers)
# L2モデルのロード
gpt2_model_ja = GPT2Model.from_pretrained(gpt2_model_japanese_name)
gpt2_model_ja_modified.load_state_dict(modified_ja_model_dict)
# test
print(gpt2_model_ja_modified.state_dict)
topk_outliers_modified = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_ja_modified, topk)
print(topk_outliers_modified.keys())

# print(modified_ja_model)
# print(type(modified_ja_model))
# # dutch
# topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_du, topk)
# modified_du_model = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_du, topk_outliers)
# # german
# topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_ger, topk)
# modified_ger_model = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_ger, topk_outliers)
# # italy
# topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_ita, topk)
# modified_ita_model = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_ita, topk_outliers)
# # french
# topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_fre, topk)
# modified_fre_model = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_fre, topk_outliers)
# # korean
# topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_ko, topk)
# modified_ko_model = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_ko, topk_outliers)
# # spanish
# topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_spa, topk)
# modified_spa_model = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_spa, topk_outliers)

""" それぞれ、modify前のモデルと、modify後のBLiMPの精度を測る """
