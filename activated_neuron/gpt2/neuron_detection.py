import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/gpt2")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from neuron_detection_funcs import *
from visualization_funcs import *

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
    # select first 100 sentences
    dataset = dataset.select(range(500))
    tatoeba_data = [(item['translation'][L1], item['translation'][L2]) for item in dataset]
    tatoeba_data_len = len(tatoeba_data)

    neuron_detection_dict = track_neurons_with_text_data(model, tokenizer, tatoeba_data, 0.5)

    activated_neurons_L1 = neuron_detection_dict["activated_neurons_L1"]
    activated_neurons_L2 = neuron_detection_dict["activated_neurons_L2"]
    non_activated_neurons_L1 = neuron_detection_dict["non_activated_neurons_L1"]
    non_activated_neurons_L2 = neuron_detection_dict["non_activated_neurons_L2"]
    shared_neurons = neuron_detection_dict["shared_neurons"]
    specific_neurons_L1 = neuron_detection_dict["specific_neurons_L1"]
    specific_neurons_L2 = neuron_detection_dict["specific_neurons_L2"]
    non_activated_neurons_all = neuron_detection_dict["non_activated_neurons_all"]

    """ visualization """
    visualize_neurons_with_line_plot(
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
                                        "tatoeba_0.5_th"
                                    )
