# import os
import sys
sys.path.append('../../proj_LA/measure_similarities')
# path = '../../proj_LA/measure_similarities'
# print(os.path.exists(path))

import torch

from collections import defaultdict
from transformers import AutoModel, GPT2Model
from datasets import load_dataset, get_dataset_config_names

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
# gpt2_model_du = GPT2Model.from_pretrained(gpt2_model_dutch_name) # du
# gpt2_model_ger = GPT2Model.from_pretrained(gpt2_model_german_name) # ger
# gpt2_model_ita = GPT2Model.from_pretrained(gpt2_model_italian_name) # ita
# gpt2_model_fre = GPT2Model.from_pretrained(gpt2_model_french_name) # fre
# gpt2_model_ko = GPT2Model.from_pretrained(gpt2_model_korean_name) # ko
# gpt2_model_spa = GPT2Model.from_pretrained(gpt2_model_spanish_name) # spa

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
print(state_dict_gpt2_original.keys() == state_dict_gpt2_ja.keys())
print(state_dict_gpt2_original.keys() == state_dict_gpt2_ja.keys())
print(state_dict_gpt2_original.keys() == state_dict_gpt2_du.keys())
print(state_dict_gpt2_original.keys() == state_dict_gpt2_ger.keys())
print(state_dict_gpt2_original.keys() == state_dict_gpt2_ita.keys())
print(state_dict_gpt2_original.keys() == state_dict_gpt2_fre.keys())
print(state_dict_gpt2_original.keys() == state_dict_gpt2_ko.keys())
print(state_dict_gpt2_original.keys() == state_dict_gpt2_spa.keys())

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
topk = 20
topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_ja, topk)
modified_L2_model = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_ja, topk_outliers)
print(topk_outliers.keys())
print('\n')
print('---------------------------------------------------------------')
print('\n')
topk_outliers_modified = get_topk_outliers_parameters(state_dict_gpt2_original, modified_L2_model, topk)
print(topk_outliers_modified.keys())

""" それぞれのL2モデルのmodified versionを作成 """
topk = 20
""" japanese """
# topkこのoutliers(base modelと比べて、一番差があるパラメータを持つ層)を取得
topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_ja, topk)
# topkこの層のパラメータに対して緩和操作を実施
modified_ja_model_dict = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_ja, topk_outliers)
# L1->L2モデルのロード
gpt2_model_ja = GPT2Model.from_pretrained(gpt2_model_japanese_name)
gpt2_model_ja_modified.load_state_dict(modified_ja_model_dict)
# test
print(gpt2_model_ja_modified.state_dict)
topk_outliers_modified = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_ja_modified, topk)
print(topk_outliers_modified.keys())

# """ dutch """
# topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_du, topk)
# modified_du_model_dict = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_du, topk_outliers)
# gpt2_model_du = GPT2Model.from_pretrained(gpt2_model_dutch_name)
# gpt2_model_du_modified.load_state_dict(modified_du_model_dict)
# """ german """
# topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_ger, topk)
# modified_ger_model_dict = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_ger, topk_outliers)
# gpt2_model_ger = GPT2Model.from_pretrained(gpt2_model_german_name)
# gpt2_model_ger_modified.load_state_dict(modified_ger_model_dict)
# """ italy """
# topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_ita, topk)
# modified_ita_model_dict = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_ita, topk_outliers)
# gpt2_model_ita = GPT2Model.from_pretrained(gpt2_model_italian_name)
# gpt2_model_ita_modified.load_state_dict(modified_ita_model_dict)
# """ french """
# topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_fre, topk)
# modified_fre_model_dict = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_fre, topk_outliers)
# gpt2_model_fre = GPT2Model.from_pretrained(gpt2_model_italian_name)
# gpt2_model_fre_modified.load_state_dict(modified_fre_model_dict)
# """ korean """
# topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_ko, topk)
# modified_ko_model_dict = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_ko, topk_outliers)
# gpt2_model_ko = GPT2Model.from_pretrained(gpt2_model_korean_name)
# gpt2_model_ko_modified.load_state_dict(modified_ko_model_dict)
# """ spanish """
# topk_outliers = get_topk_outliers_parameters(state_dict_gpt2_original, state_dict_gpt2_spa, topk)
# modified_spa_model_dict = relax_abs_diff_of_L2_model(state_dict_gpt2_original, state_dict_gpt2_spa, topk_outliers)
# gpt2_model_spa = GPT2Model.from_pretrained(gpt2_model_spanish_name)
# gpt2_model_spa_modified.load_state_dict(modified_spa_model_dict)

""" それぞれ、modify前のモデルと、modify後のBLiMPの精度を測る """
# BLiMPの評価項目リスト
configs = get_dataset_config_names("blimp")

def evaluate_model_on_blimp(model, config):
    # BliMPデータセットを読み込む
    blimp_dataset = load_dataset("blimp")

    # テストデータをデータローダーでバッチ処理
    test_loader = DataLoader(blimp_dataset['test'], batch_size=config['batch_size'])

    model.eval()  # モデルを評価モードにする
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 勾配計算を無効にする
        for batch in test_loader:
            inputs = batch['input_ids'].to(config['device'])  # 入力をデバイスに移動
            labels = batch['labels'].to(config['device'])  # ラベルをデバイスに移動

            outputs = model(inputs)  # モデルの出力を取得
            logits = outputs.logits  # ロジットを取得
            preds = torch.argmax(logits, dim=-1)  # 最も高いロジットを持つクラスを予測

            all_preds.extend(preds.cpu().numpy())  # CPUに戻してからリストに追加
            all_labels.extend(labels.cpu().numpy())

    # 精度を計算
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# BLiMPデータセットを評価し、結果を保存する関数
def evaluate_models_on_blimp(models, configs):
    results = defaultdict(dict)

    for config in configs:
        # 各モデルに対してBLiMPデータセットを評価
        for model_name, model in models.items():
            accuracy = evaluate_model_on_blimp(model, config)  # 評価関数は適宜定義する必要があります
            results[config][model_name] = accuracy

    return results

# モデルを辞書として格納
models = {
    "Original Japanese Model": gpt2_model_ja,
    "Modified Japanese Model": gpt2_model_ja_modified,
    # "Original Dutch Model": gpt2_model_du,
    # "Modified Dutch Model": gpt2_model_du_modified,
    # "Original German Model": gpt2_model_ger,
    # "Modified German Model": gpt2_model_ger_modified,
    # "Original Italian Model": gpt2_model_ita,
    # "Modified Italian Model": gpt2_model_ita_modified,
    # "Original French Model": gpt2_model_fre,
    # "Modified French Model": gpt2_model_fre_modified,
    # "Original Korean Model": gpt2_model_ko,
    # "Modified Korean Model": gpt2_model_ko_modified,
    # "Original Spanish Model": gpt2_model_spa,
    # "Modified Spanish Model": gpt2_model_spa_modified,
}

# BLiMPデータセットを評価
blimp_results = evaluate_models_on_blimp(models, configs)

# 結果をDataFrameに変換
results_df = pd.DataFrame(blimp_results)

# 各文法項目に対する各モデルの正答率を表示
print(results_df)

# 各fieldごとに正答率を計算
field_accuracies = results_df.mean(axis=0)  # 各列（モデル）ごとの平均を計算

# 各fieldごとの結果もDataFrameに追加
results_df.loc['Field Average'] = field_accuracies

# DataFrameをCSVファイルに保存
# results_df.to_csv('blimp_results.csv', index=True)

# print("Results saved to blimp_results.csv")
