import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# CSVファイルの読み込み
df = pd.read_csv('/home/s2410121/proj_LA/gpt2-small_blimp/csv_files_final/blimp_evaluation_results_complete2_llama3_all_final.csv')

# LLaMA-3
llama_model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B",
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    "nl": "ReBatch/Llama-3-8B-dutch", # du
    "it": "DeepMount00/Llama-3-8b-Ita", # ita
    "ko": "beomi/Llama-3-KoEn-8B", # ko
}

""" 全モデルを同時にプロット """
# タスクごとの精度をプロット（モデル別で色分け）
plt.figure(figsize=(10, 8))
sns.barplot(x='Task', y='Accuracy', hue='Model', data=df)
plt.xticks(rotation=90)
plt.title('Average Accuracy per Task by Model')
plt.xlabel('Task')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('/home/s2410121/proj_LA/gpt2-small_blimp/images/llama3/barplot.png')
plt.close()

""" 各モデルをbase modelと比較 """
# ベースモデル（meta-llama/Meta-Llama-3-8B）のデータを抽出
base_model = df[df['Model'] == 'meta-llama/Meta-Llama-3-8B']

# 他のモデルのデータを抽出
other_models = df[df['Model'] != 'meta-llama/Meta-Llama-3-8B']

# ベースモデルと他のモデルの精度差を計算
# 他のモデルとベースモデルの精度をタスクごとに比較
comparison_df = other_models.merge(base_model[['Task', 'Accuracy']], on='Task', suffixes=('_other', '_base'))

# 精度差を計算
comparison_df['Accuracy_diff'] = comparison_df['Accuracy_other'] - comparison_df['Accuracy_base']

# 可視化
plt.figure(figsize=(12, 8))
sns.barplot(data=comparison_df, x='Task', y='Accuracy_diff', hue='Model', palette='Set2')

plt.title('Model Accuracy Comparison')
plt.xlabel('Task')
plt.ylabel('Accuracy Difference')
plt.xticks(rotation=90)
plt.tight_layout()

# グラフを保存
plt.savefig('/home/s2410121/proj_LA/gpt2-small_blimp/images/llama3/task_accuracy_comparison.png')
plt.close()


"""  """
