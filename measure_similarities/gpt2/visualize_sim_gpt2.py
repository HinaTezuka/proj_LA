""" gpt2-smallの類似度・差異計算結果の可視化 """

from gpt2_measure_similarity import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


""" bar plot """

# 各モデルの平均絶対差を計算
en_ja_mean_diff = np.mean([values['mean_absolute_difference'] for values in weight_changes_gpt2['en_ja'].values()])
en_du_mean_diff = np.mean([values['mean_absolute_difference'] for values in weight_changes_gpt2['en_du'].values()])
en_ger_mean_diff = np.mean([values['mean_absolute_difference'] for values in weight_changes_gpt2['en_ger'].values()])
en_ita_mean_diff = np.mean([values['mean_absolute_difference'] for values in weight_changes_gpt2['en_ita'].values()])
en_fre_mean_diff = np.mean([values['mean_absolute_difference'] for values in weight_changes_gpt2['en_fre'].values()])
en_ko_mean_diff = np.mean([values['mean_absolute_difference'] for values in weight_changes_gpt2['en_ko'].values()])
en_spa_mean_diff = np.mean([values['mean_absolute_difference'] for values in weight_changes_gpt2['en_spa'].values()])

# 各モデルのコサイン類似度を計算
en_ja_cos_sim = np.mean([values['cos_sim'] for values in weight_changes_gpt2['en_ja'].values()])
en_du_cos_sim = np.mean([values['cos_sim'] for values in weight_changes_gpt2['en_du'].values()])
en_ger_cos_sim = np.mean([values['cos_sim'] for values in weight_changes_gpt2['en_ger'].values()])
en_ita_cos_sim = np.mean([values['cos_sim'] for values in weight_changes_gpt2['en_ita'].values()])
en_fre_cos_sim = np.mean([values['cos_sim'] for values in weight_changes_gpt2['en_fre'].values()])
en_ko_cos_sim = np.mean([values['cos_sim'] for values in weight_changes_gpt2['en_ko'].values()])
en_spa_cos_sim = np.mean([values['cos_sim'] for values in weight_changes_gpt2['en_spa'].values()])

# 結果をデータフレームにまとめる
mean_diffs = pd.DataFrame({
    'Model': ['en_ja', 'en_du', 'en_ger', 'en_ita', 'en_fre', 'en_ko', 'en_spa'],
    'Mean Absolute Difference': [en_ja_mean_diff, en_du_mean_diff, en_ger_mean_diff, en_ita_mean_diff, en_fre_mean_diff, en_ko_mean_diff, en_spa_mean_diff],
    'Cosine Similarity': [en_ja_cos_sim, en_du_cos_sim, en_ger_cos_sim, en_ita_cos_sim, en_fre_cos_sim, en_ko_cos_sim, en_spa_cos_sim]
})

# 可視化（平均絶対差）
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(mean_diffs['Model'], mean_diffs['Mean Absolute Difference'], color=['blue', 'orange', 'red', 'green', 'black', 'yellow', 'gray'])
plt.xlabel('Model')
plt.ylabel('Mean Absolute Difference')
plt.title('Comparison of Mean Absolute Differences')
plt.ylim(0, max(mean_diffs['Mean Absolute Difference']) * 1.1)  # Y軸の範囲を調整
plt.grid(axis='y')

# 可視化（コサイン類似度）
plt.subplot(1, 2, 2)
plt.bar(mean_diffs['Model'], mean_diffs['Cosine Similarity'], color=['blue', 'orange', 'red', 'green', 'black', 'yellow', 'gray'])
plt.xlabel('Model')
plt.ylabel('Cosine Similarity')
plt.title('Comparison of Cosine Similarity')
plt.ylim(0, 1)  # コサイン類似度は0から1の範囲
plt.grid(axis='y')

plt.tight_layout()  # レイアウトを調整
plt.show()


""" layerごとの絶対差の変遷のplot """

# 各モデルの平均絶対差を計算
mean_diffs = {
    'en_ja': [values['mean_absolute_difference'] for values in weight_changes_gpt2['en_ja'].values()],
    'en_du': [values['mean_absolute_difference'] for values in weight_changes_gpt2['en_du'].values()],
    'en_ger': [values['mean_absolute_difference'] for values in weight_changes_gpt2['en_ger'].values()],
    'en_ita': [values['mean_absolute_difference'] for values in weight_changes_gpt2['en_ita'].values()],
    'en_fre': [values['mean_absolute_difference'] for values in weight_changes_gpt2['en_fre'].values()],
    'en_ko': [values['mean_absolute_difference'] for values in weight_changes_gpt2['en_ko'].values()],
    'en_spa': [values['mean_absolute_difference'] for values in weight_changes_gpt2['en_spa'].values()],
}

# 各リストの最小の長さを取得
min_length = min(len(mean_diffs[model]) for model in mean_diffs)

# 各モデルのリストを最小の長さに揃える
for model in mean_diffs:
    mean_diffs[model] = mean_diffs[model][:min_length]

# DataFrameに変換
mean_diffs_df = pd.DataFrame(mean_diffs)

# 可視化 (折れ線グラフ)
plt.figure(figsize=(12, 6))
for model in mean_diffs_df.columns:
    plt.plot(mean_diffs_df.index, mean_diffs_df[model], marker='o', label=model)

plt.xlabel('Layer Index')
plt.ylabel('Mean Absolute Difference')
plt.title('Mean Absolute Differences Across Models')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
