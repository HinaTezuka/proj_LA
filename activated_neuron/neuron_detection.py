""" neurons detection """

import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron")

import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from neuron_detection_funcs import track_neurons_with_text_data
from visualization_funcs import visualize_neurons_with_line_plot, visualize_neurons_with_line_plot_simple, visualize_neurons_with_line_plot_b

""" models """
# LLaMA-3
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B",
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    "nl": "ReBatch/Llama-3-8B-dutch", # du
    "it": "DeepMount00/Llama-3-8b-Ita", # ita
    "ko": "beomi/Llama-3-KoEn-8B", # ko
}

L1 = "en" # L1 is fixed to english.

for L2, model_name in model_names.items():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    # model = AutoModelForCausalLM.from_pretrained(model_name)

    """ tatoeba translation corpus """
    dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
    # select first 100 sentences
    num_sentences = 2000
    dataset = dataset.select(range(num_sentences))
    tatoeba_data = []
    for item in dataset:
        # check if there are empty sentences.
        if item['translation'][L1] != '' and item['translation'][L2] != '':
            tatoeba_data.append((item['translation'][L1], item['translation'][L2]))
    # tatoeba_data = [(item['translation'][L1], item['translation'][L2]) for item in dataset]
    tatoeba_data_len = len(tatoeba_data)
    # print(tatoeba_data_len)

    """ baseとして、対訳関係のない1文ずつのペアを作成(tatoebaの最後の1, 2文が対象) """
    tatoeba_data_base = [(dataset["translation"][1][L1], dataset["translation"][5][L2])]
    print(tatoeba_data_base)

    """ tracking neurons """
    neuron_detection_dict, neuron_detection_dict_vis = track_neurons_with_text_data(model, 'llama', tokenizer, tatoeba_data, 0.1, 0)
    neuron_detection_base_dict, neuron_detection_base_dict_vis = track_neurons_with_text_data(model, 'llama', tokenizer, tatoeba_data_base, 0.1, 0)
    # print(len(neuron_detection_base_dict["shared_neurons"]))
    # print(len(neuron_detection_dict["shared_neurons"]))
    # sys.exit()

    # delete some cache
    del model
    torch.cuda.empty_cache()
    """ main """
    #
    activated_neurons_L1 = neuron_detection_dict["activated_neurons_L1"]
    activated_neurons_L2 = neuron_detection_dict["activated_neurons_L2"]
    non_activated_neurons_L1 = neuron_detection_dict["non_activated_neurons_L1"]
    non_activated_neurons_L2 = neuron_detection_dict["non_activated_neurons_L2"]
    shared_neurons = neuron_detection_dict["shared_neurons"]
    specific_neurons_L1 = neuron_detection_dict["specific_neurons_L1"]
    specific_neurons_L2 = neuron_detection_dict["specific_neurons_L2"]
    non_activated_neurons_all = neuron_detection_dict["non_activated_neurons_all"]
    # for visualization
    activated_neurons_L1_vis = neuron_detection_dict_vis["activated_neurons_L1"]
    activated_neurons_L2_vis = neuron_detection_dict_vis["activated_neurons_L2"]
    non_activated_neurons_L1_vis = neuron_detection_dict_vis["non_activated_neurons_L1"]
    non_activated_neurons_L2_vis = neuron_detection_dict_vis["non_activated_neurons_L2"]
    shared_neurons_vis = neuron_detection_dict_vis["shared_neurons"]
    specific_neurons_L1_vis = neuron_detection_dict_vis["specific_neurons_L1"]
    specific_neurons_L2_vis = neuron_detection_dict_vis["specific_neurons_L2"]
    non_activated_neurons_all_vis = neuron_detection_dict_vis["non_activated_neurons_all"]

    """ for base line """
    activated_neurons_L1_base = neuron_detection_base_dict["activated_neurons_L1"]
    activated_neurons_L2_base = neuron_detection_base_dict["activated_neurons_L2"]
    non_activated_neurons_L1_base = neuron_detection_base_dict["non_activated_neurons_L1"]
    non_activated_neurons_L2_base = neuron_detection_base_dict["non_activated_neurons_L2"]
    shared_neurons_base = neuron_detection_base_dict["shared_neurons"]
    specific_neurons_L1_base = neuron_detection_base_dict["specific_neurons_L1"]
    specific_neurons_L2_base = neuron_detection_base_dict["specific_neurons_L2"]
    non_activated_neurons_all_base = neuron_detection_base_dict["non_activated_neurons_all"]
    # for visualization
    activated_neurons_L1_base_vis = neuron_detection_base_dict_vis["activated_neurons_L1"]
    activated_neurons_L2_base_vis = neuron_detection_base_dict_vis["activated_neurons_L2"]
    non_activated_neurons_L1_base_vis = neuron_detection_base_dict_vis["non_activated_neurons_L1"]
    non_activated_neurons_L2_base_vis = neuron_detection_base_dict_vis["non_activated_neurons_L2"]
    shared_neurons_base_vis = neuron_detection_base_dict_vis["shared_neurons"]
    specific_neurons_L1_base_vis = neuron_detection_base_dict_vis["specific_neurons_L1"]
    specific_neurons_L2_base_vis = neuron_detection_base_dict_vis["specific_neurons_L2"]
    non_activated_neurons_all_base_vis = neuron_detection_base_dict_vis["non_activated_neurons_all"]

    """ visualization """
    visualize_neurons_with_line_plot_simple(
                                        L1,
                                        L2,
                                        # main
                                        activated_neurons_L1_vis,
                                        activated_neurons_L2_vis,
                                        non_activated_neurons_L1_vis,
                                        non_activated_neurons_L2_vis,
                                        shared_neurons_vis,
                                        specific_neurons_L1_vis,
                                        specific_neurons_L2_vis,
                                        non_activated_neurons_all_vis,
                                        "tatoeba_0.1_th",
                                        # base line
                                        shared_neurons_base_vis,
                                    )

    # visualize_neurons_with_line_plot(
    #                                     L1,
    #                                     L2,
    #                                     # main
    #                                     activated_neurons_L1,
    #                                     activated_neurons_L2,
    #                                     non_activated_neurons_L1,
    #                                     non_activated_neurons_L2,
    #                                     shared_neurons,
    #                                     specific_neurons_L1,
    #                                     specific_neurons_L2,
    #                                     non_activated_neurons_all,
    #                                     "tatoeba",
    #                                     # base line
    #                                     shared_neurons_base,
    #                                 )

    # visualize_neurons_with_line_plot_b(
    #                                     L1,
    #                                     L2,
    #                                     # base line
    #                                     activated_neurons_L1_base_vis,
    #                                     activated_neurons_L2_base_vis,
    #                                     non_activated_neurons_L1_base_vis,
    #                                     non_activated_neurons_L2_base_vis,
    #                                     shared_neurons_base_vis,
    #                                     specific_neurons_L1_base_vis,
    #                                     specific_neurons_L2_base_vis,
    #                                     non_activated_neurons_all_base_vis,
    #                                     "only_1s"
    #                                 )

if __name__ == "__main__":
    print('visualization completed.')

    # for layer_idx, neurons in activated_neurons_L1:
    #     print(f"Layer {layer_idx}: Shared Neurons: {neurons}")
    # print(f'len_of_nonactivated_neurons_L2: {len(non_activated_neurons_L2)}, len_of_non_activated_neurons_L1: {len(non_activated_neurons_L1)}')
    # print(f'non_activated_neurons_all: {non_activated_neurons_all}')
