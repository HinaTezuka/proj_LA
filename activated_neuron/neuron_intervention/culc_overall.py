""" overallをみる """

import pandas as pd

# CSVファイルを読み込んでデータフレームを作成
df = pd.read_csv("/home/s2410121/proj_LA/activated_neuron/neuron_intervention/csv_files/jblimp/shared/all/llama3_en_ja_shared_non_translation.csv")

# 各モデルごとに正答率の平均を計算します
overall_accuracy = df.groupby('Model')['Accuracy'].mean().reset_index()

# 列名を変更してOVERALLにします
overall_accuracy.rename(columns={'Accuracy': 'OVERALL'}, inplace=True)

# 結果を表示
print(overall_accuracy)
