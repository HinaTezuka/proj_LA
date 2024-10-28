import pandas as pd

from gpt2_visualize_blimp_eval_funcs import *

# CSVファイルをDataFrameとして読み込み
file_path = "/home/s2410121/proj_LA/gpt2-small_blimp/blimp_evaluation_results_test.csv"
# CSVファイルを読み込む
# データの読み込み
data = pd.read_csv(file_path)

acc_comparison(data)
multiple_models_acc_comparison(data)
models_above_base_model(data)
