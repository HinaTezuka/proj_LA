import os
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/detect_act_neurons.py")
sys.path.append("/home/s2410121/proj_LA/activated_neuron/shared_neurons.py")

import matplotlib.pyplot as plt

def visualize_neurons_with_line_plot(
                                    L1,
                                    L2,
                                    activated_neurons_L1,
                                    activated_neurons_L2,
                                    non_activated_neurons_L1,
                                    non_activated_neurons_L2,
                                    shared_neurons,
                                    specific_neurons_L1,
                                    specific_neurons_L2,
                                    non_activated_neurons_all
                                    ):
    # 層の数
    num_layers = 32

    # 各層のニューロンの活性化を集計するためのリスト
    L2_counts = [0] * num_layers
    L1_counts = [0] * num_layers
    shared_counts = [0] * num_layers
    specific_L2_counts = [0] * num_layers
    specific_L1_counts = [0] * num_layers
    non_activated_L2_counts = [0] * num_layers  # 日本語の非活性化ニューロン
    non_activated_L1_counts = [0] * num_layers  # 英語の非活性化ニューロン
    non_activated_all_counts = [0] * num_layers  # 両言語の非活性化共通ニューロン

    # 各層におけるニューロンの活性化数をカウント
    for layer_idx in range(num_layers):
        L2_counts[layer_idx] = len(activated_neurons_L2[layer_idx][1])  # 日本語のニューロン数
        L1_counts[layer_idx] = len(activated_neurons_L1[layer_idx][1])    # 英語のニューロン数
        shared_counts[layer_idx] = len(shared_neurons[layer_idx][1])                # 共有ニューロン数
        specific_L2_counts[layer_idx] = len(specific_neurons_L2[layer_idx][1])  # 日本語特有のニューロン数
        specific_L1_counts[layer_idx] = len(specific_neurons_L1[layer_idx][1])    # 英語特有のニューロン数
        non_activated_L2_counts[layer_idx] = len(non_activated_neurons_L2[layer_idx][1])  # 日本語の非活性化ニューロン数
        non_activated_L1_counts[layer_idx] = len(non_activated_neurons_L1[layer_idx][1])    # 英語の非活性化ニューロン数
        non_activated_all_counts[layer_idx] = len(non_activated_neurons_all[layer_idx][1])  # 両言語の非活性化共通ニューロン数

    # プロット
    plt.figure(figsize=(15, 10))
    plt.plot(range(num_layers), L2_counts, label=f'{L2} Activated Neurons', marker='o')
    plt.plot(range(num_layers), L1_counts, label=f'{L1} Activated Neurons', marker='o')
    plt.plot(range(num_layers), shared_counts, label='Shared Neurons', marker='o', linewidth=6)
    plt.plot(range(num_layers), specific_L2_counts, label=f'Specific to {L2}', marker='o')
    plt.plot(range(num_layers), specific_L1_counts, label=f'Specific to {L1}', marker='o')
    plt.plot(range(num_layers), non_activated_L2_counts, label=f'Non-Activated {L2} Neurons', marker='x', linestyle='--')
    plt.plot(range(num_layers), non_activated_L1_counts, label=f'Non-Activated {L1} Neurons', marker='x', linestyle='--')
    plt.plot(range(num_layers), non_activated_all_counts, label='Non-Activated Neurons (Both)', marker='s', linestyle='-.')

    plt.title(f'Neuron Activation Counts per Layer ({L1} and {L2})')
    plt.xlabel('Layer Index')
    plt.ylabel('Number of Neurons')
    plt.xticks(range(num_layers))
    plt.legend()
    plt.grid()

    # グラフの保存
    plt.savefig(f'/home/s2410121/proj_LA/activated_neuron/images/activated_neuron_{L1}_{L2}.png')
    plt.close()

if __name__ == "__main__":
    # visualize_neurons_with_line_plot(L1, "ja")
    print("visualization completed ! ")
