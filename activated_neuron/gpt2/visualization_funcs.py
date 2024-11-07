import os
import sys

import matplotlib.pyplot as plt

nums_of_neurons_llama3 = 3072 # nums of all neurons (GPT2-small: MLP)

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
                                    non_activated_neurons_all,
                                    folder: str,
                                    ):
    # nums of all layers(LLaMA-3-8B)
    num_layers = 12

    # list for aggregating all types of neurons
    L2_counts = [0] * num_layers
    L1_counts = [0] * num_layers
    shared_counts = [0] * num_layers
    specific_L2_counts = [0] * num_layers
    specific_L1_counts = [0] * num_layers
    non_activated_L2_counts = [0] * num_layers  # 日本語の非活性化ニューロン
    non_activated_L1_counts = [0] * num_layers  # 英語の非活性化ニューロン
    non_activated_all_counts = [0] * num_layers  # 両言語の非活性化共通ニューロン

    # counting activate/non-activate counts
    for layer_idx in range(num_layers):
        # activated_neurons_L1 と activated_neurons_L2 から、各層のユニークなニューロン数を取得
        L1_neurons_set = set(activated_neurons_L1[layer_idx][1][:, 2])  # 第3列がニューロンインデックス
        L2_neurons_set = set(activated_neurons_L2[layer_idx][1][:, 2])

        L2_counts[layer_idx] = len(L2_neurons_set)  # nums for ja neurons
        L1_counts[layer_idx] = len(L1_neurons_set)  # nums for en neurons
        shared_counts[layer_idx] = len(set(shared_neurons[layer_idx][1].flatten()))  # shared_neurons
        specific_L2_counts[layer_idx] = len(set(specific_neurons_L2[layer_idx][1].flatten()))  # specific neurons for ja
        specific_L1_counts[layer_idx] = len(set(specific_neurons_L1[layer_idx][1].flatten()))  # specific neurons for en
        # non_activated_L2_counts[layer_idx] = len(set(non_activated_neurons_L2[layer_idx][1].flatten()))  # non-activate ja
        # non_activated_L1_counts[layer_idx] = len(set(non_activated_neurons_L1[layer_idx][1].flatten()))  # non-activate en
        # non_activated_all_counts[layer_idx] = len(set(non_activated_neurons_all[layer_idx][1].flatten()))  # non-activate for both

    # plot
    plt.figure(figsize=(15, 10))
    plt.plot(range(num_layers), L2_counts, label=f'{L2} Activated Neurons', marker='o')
    plt.plot(range(num_layers), L1_counts, label=f'{L1} Activated Neurons', marker='o')
    plt.plot(range(num_layers), shared_counts, label='Shared Neurons', marker='o', linewidth=6)
    plt.plot(range(num_layers), specific_L2_counts, label=f'Specific to {L2}', marker='o')
    plt.plot(range(num_layers), specific_L1_counts, label=f'Specific to {L1}', marker='o')
    # plt.plot(range(num_layers), non_activated_L2_counts, label=f'Non-Activated {L2} Neurons', marker='x', linestyle='--')
    # plt.plot(range(num_layers), non_activated_L1_counts, label=f'Non-Activated {L1} Neurons', marker='x', linestyle='--')
    # plt.plot(range(num_layers), non_activated_all_counts, label='Non-Activated Neurons (Both)', marker='s', linestyle='-.')

    plt.title(f'Neuron Activation Counts per Layer ({L1} and {L2})')
    plt.xlabel('Layer Index')
    plt.ylabel('Number of Neurons')
    plt.xticks(range(num_layers))
    plt.legend()
    plt.grid()

    # グラフの保存
    plt.savefig(f'/home/s2410121/proj_LA/activated_neuron/gpt2/images/{folder}/activated_neuron_{L1}_{L2}.png')
    plt.close()
