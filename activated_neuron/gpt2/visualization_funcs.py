import os
import sys

import matplotlib.pyplot as plt

nums_of_neurons_llama3 = 3072 # nums of all neurons (GPT2-small: MLP)

def visualize_neurons_with_line_plot_simple(
                                    L1,
                                    L2,
                                    # main
                                    activated_neurons_L1,
                                    activated_neurons_L2,
                                    non_activated_neurons_L1,
                                    non_activated_neurons_L2,
                                    shared_neurons,
                                    specific_neurons_L1,
                                    specific_neurons_L2,
                                    non_activated_neurons_all,
                                    folder: str,
                                    # base
                                    shared_neurons_base,
                                    ):
    # nums of all layers(GPT2-small)
    num_layers = 12

    """ main """
    L2_counts = [0] * num_layers
    L1_counts = [0] * num_layers
    shared_counts = [0] * num_layers
    specific_L2_counts = [0] * num_layers
    specific_L1_counts = [0] * num_layers
    non_activated_L2_counts = [0] * num_layers  # 日本語の非活性化ニューロン
    non_activated_L1_counts = [0] * num_layers  # 英語の非活性化ニューロン
    non_activated_all_counts = [0] * num_layers  # 両言語の非活性化共通ニューロン
    """ base line """
    shared_counts_base = [0] * num_layers

    # counting activate/non-activate counts
    for layer_idx in range(num_layers):

        L1_counts[layer_idx] = len(activated_neurons_L1[layer_idx])  # nums for ja neurons
        L2_counts[layer_idx] = len(activated_neurons_L2[layer_idx])  # nums for en neurons
        shared_counts[layer_idx] = len(shared_neurons[layer_idx])  # shared_neurons
        specific_L2_counts[layer_idx] = len(specific_neurons_L2[layer_idx])  # specific neurons for ja
        specific_L1_counts[layer_idx] = len(specific_neurons_L1[layer_idx])  # specific neurons for en
        # non_activated_L2_counts[layer_idx] = len(non_activated_neurons_L2[layer_idx])  # non-activate ja
        # non_activated_L1_counts[layer_idx] = len(non_activated_neurons_L1[layer_idx])  # non-activate en
        # non_activated_all_counts[layer_idx] = len(non_activated_neurons_all[layer_idx])  # non-activate for both
        """ base line """
        shared_counts_base[layer_idx] = len(shared_neurons_base[layer_idx])  # shared_neurons
    # plot
    plt.figure(figsize=(15, 10))
    plt.plot(range(num_layers), L2_counts, label=f'{L2} Activated Neurons', marker='o')
    plt.plot(range(num_layers), L1_counts, label=f'{L1} Activated Neurons', marker='o')
    plt.plot(range(num_layers), shared_counts, label='Shared Neurons(sentence pair of same meanings)', marker='o', linewidth=6)
    plt.plot(range(num_layers), specific_L2_counts, label=f'Specific to {L2}', marker='o')
    plt.plot(range(num_layers), specific_L1_counts, label=f'Specific to {L1}', marker='o')
    # plt.plot(range(num_layers), non_activated_L2_counts, label=f'Non-Activated {L2} Neurons', marker='x', linestyle='--')
    # plt.plot(range(num_layers), non_activated_L1_counts, label=f'Non-Activated {L1} Neurons', marker='x', linestyle='--')
    # plt.plot(range(num_layers), non_activated_all_counts, label='Non-Activated Neurons (Both)', marker='s', linestyle='-.')
    """ base line """
    plt.plot(range(num_layers), shared_counts_base, label='Shared Neurons(base)', marker='x', linewidth=2)

    plt.title(f'Neuron Activation Counts per Layer ({L1} and {L2})')
    plt.xlabel('Layer Index')
    plt.ylabel('Number of Neurons')
    plt.xticks(range(num_layers))
    plt.legend()
    plt.grid()

    # グラフの保存
    plt.savefig(f'/home/s2410121/proj_LA/activated_neuron/gpt2/images/{folder}/activated_neuron_{L1}_{L2}.png')
    plt.close()
