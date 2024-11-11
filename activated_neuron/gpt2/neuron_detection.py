import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/gpt2")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from neuron_detection_funcs import track_neurons_with_text_data
from visualization_funcs import visualize_neurons_with_line_plot_simple

# GPT-2-small
model_names = {
    # "base": "gpt2",
    "ja": "rinna/japanese-gpt2-small", # ja
    # "de": "ml6team/gpt2-small-german-finetune-oscar", # ger
    "nl": "GroNLP/gpt2-small-dutch", # du
    "it": "GroNLP/gpt2-small-italian", # ita
    "fr": "dbddv01/gpt2-french-small", # fre
    "ko": "skt/kogpt2-base-v2", # ko
    "es": "datificate/gpt2-small-spanish" # spa
}

L1 = "en" # L1 is fixed to english

for L2, model_name in model_names.items():
    """ load model and tokenizer """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    """ tatoeba translation corpus """
    # Dataset({
    #     features: ['id', 'translation'],
    #     num_rows: 208866
    # })
    dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
    # select first __ sentences
    num_sentences = 2000
    dataset = dataset.select(range(num_sentences))
    tatoeba_data = [(item['translation'][L1], item['translation'][L2]) for item in dataset]
    tatoeba_data_len = len(tatoeba_data)
    """ baseとして、対訳関係のない1文ずつのペアを作成(tatoebaの最後の1, 2文が対象) """
    tatoeba_data_base = [(dataset["translation"][1][L1], dataset["translation"][5][L2])]
    print(tatoeba_data_base)

    neuron_detection_dict, neuron_detection_dict_vis = track_neurons_with_text_data(model, 'gpt2', tokenizer, tatoeba_data, 0.1, 0)
    _ , neuron_detection_base_dict_vis = track_neurons_with_text_data(model, 'gpt2', tokenizer, tatoeba_data_base, 0.1, 0)

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
    shared_neurons_base_vis = neuron_detection_base_dict_vis["shared_neurons"]

    """ visualization """
    visualize_neurons_with_line_plot_simple(
                                            L1,
                                            L2,
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
                                            shared_neurons_base_vis
                                           )
