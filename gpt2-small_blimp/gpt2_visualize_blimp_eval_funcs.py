import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルをDataFrameとして読み込み
file_path = "/home/s2410121/proj_LA/gpt2-small_blimp/blimp_evaluation_results_test.csv"  # 実際のCSVファイルのパスに置き換えてください
# CSVファイルを読み込む
# データの読み込み
data = pd.read_csv(file_path)
# df = pd.read_csv(file_path)  # 適切なファイルパスに置き換えてください

""" 各モデルのaccuracyを単純に比較 """
def acc_comparison(df) -> None:
    # モデルごとにフィルタリングして可視化
    models = data['Model'].unique()
    fig, ax = plt.subplots(figsize=(15, 10))

    for model in models:
        subset = data[data['Model'] == model]
        ax.barh(subset['Task'], subset['Accuracy'], label=model, alpha=0.7)

    # グラフの装飾
    ax.set_xlabel('Accuracy')
    ax.set_title('Task-wise Accuracy Comparison across Models')
    ax.legend(title='Model')

    # 画像ファイルとして保存
    plt.savefig('images/gpt2/accuracy_comparison_gpt2_en_ja_du.png', bbox_inches='tight')


""" 各モデルごと """
def multiple_models_acc_comparison(df) -> None:
    models = data['Model'].unique()
    # サブプロットの作成
    num_models = len(models)
    fig, axes = plt.subplots(num_models, 1, figsize=(15, 5 * num_models), sharex=True)

    for ax, model in zip(axes, models):
        subset = data[data['Model'] == model]
        ax.barh(subset['Task'], subset['Accuracy'], alpha=0.7)
        ax.set_title(model)
        ax.set_xlabel('Accuracy')

    # グラフの装飾
    plt.suptitle('Task-wise Accuracy Comparison across Models', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # サブプロットのタイトルのスペースを調整

    # 画像ファイルとして保存
    plt.savefig('images/gpt2/multiple_models_accuracy_comparison_gpt2_en_ja_du.png', bbox_inches='tight')


""" 一番性能の良かったモデルのみを表示 """
def models_above_base_model(df) -> None:
    # GPT-2の精度を取得
    baseline_accuracy = data.loc[data['Model'] == 'gpt2', 'Accuracy'].values[0]
    print(baseline_accuracy)

    # GPT-2を上回るモデルがあるタスクを特定
    tasks_with_above_baseline = data[data['Accuracy'] > baseline_accuracy]['Task'].unique()

    # 可視化するためのデータをフィルタリング
    filtered_data = data[data['Task'].isin(tasks_with_above_baseline) & (data['Model'] != 'gpt2')]

    # モデルごとにフィルタリングして可視化
    models = filtered_data['Model'].unique()
    fig, ax = plt.subplots(figsize=(15, 10))

    for model in models:
        subset = filtered_data[filtered_data['Model'] == model]
        ax.barh(subset['Task'], subset['Accuracy'], label=model, alpha=0.7)

    # 各モデルがGPT-2を上回ったタスクの数をカウント
    task_count = len(tasks_with_above_baseline)
    above_baseline_counts = {model: 0 for model in models}

    # GPT-2のカウントを追加
    gpt2_count = 0
    for task in tasks_with_above_baseline:
        if any(filtered_data[(filtered_data['Task'] == task) & (filtered_data['Model'] != 'gpt2')]['Accuracy'] > baseline_accuracy):
            best_model = filtered_data[filtered_data['Task'] == task].sort_values(by='Accuracy', ascending=False).iloc[0]
            above_baseline_counts[best_model['Model']] += 1
        if baseline_accuracy > filtered_data[filtered_data['Task'] == task]['Accuracy'].max():
            gpt2_count += 1

    # 各モデルのタスク数を割合として計算
    above_baseline_ratios = {model: (count / task_count * 100) for model, count in above_baseline_counts.items()}

    # GPT-2の割合も追加
    above_baseline_ratios['gpt2'] = (gpt2_count / task_count * 100)

    # グラフの装飾
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Task-wise Accuracy Comparison (Models Outperforming GPT-2)', fontsize=16)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

    # モデルがGPT-2を上回る性能を発揮したタスクの割合をグラフの下に表示
    plt.figtext(0.5, 0.01,
                '\n'.join([f'{model}: {above_baseline_ratios[model]:.2f}% of tasks' for model in above_baseline_ratios]),
                ha='center', fontsize=10)

    # 画像ファイルとして保存
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 下部のマージンを調整
    plt.savefig('images/gpt2/models_above_gpt2_task_comparison_with_task_ratios_gpt2_en_ja_du.png', bbox_inches='tight')
