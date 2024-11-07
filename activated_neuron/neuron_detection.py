""" neurons detection """

import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron")

import numpy as np
import matplotlib.pyplot as plt
import torch

from baukit import Trace, TraceDict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModel

import neuron_detection_funcs
import visualization_funcs

""" models """
# LLaMA-3
model_names = {
    # "base": "meta-llama/Meta-Llama-3-8B"
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
    # "de": "DiscoResearch/Llama3-German-8B", # ger
    # "nl": "ReBatch/Llama-3-8B-dutch", # du
    # "it": "DeepMount00/Llama-3-8b-Ita", # ita
    # "ko": "beomi/Llama-3-KoEn-8B", # ko
}

L1 = "en" # L1 is fixed to english.

for L2, model_name in model_names.items():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    # model = AutoModelForCausalLM.from_pretrained(model_name)

    """ tatoeba translation corpus """
    # Dataset({
    #     features: ['id', 'translation'],
    #     num_rows: 208866
    # })
    dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
    # select first 100 sentences
    dataset = dataset.select(range(100))
    tatoeba_data = [(item['translation'][L1], item['translation'][L2]) for item in dataset]
    tatoeba_data_len = len(tatoeba_data)

    """ tracking neurons """
    neuron_detection_dict = neuron_detection_funcs.track_neurons_with_text_data(model, tokenizer, tatoeba_data, 0, 0)

    # delete some cache
    # del model
    # torch.cuda.empty_cache()

    activated_neurons_L1 = neuron_detection_dict["activated_neurons_L1"]
    activated_neurons_L2 = neuron_detection_dict["activated_neurons_L2"]
    non_activated_neurons_L1 = neuron_detection_dict["non_activated_neurons_L1"]
    non_activated_neurons_L2 = neuron_detection_dict["non_activated_neurons_L2"]
    shared_neurons = neuron_detection_dict["shared_neurons"]
    specific_neurons_L1 = neuron_detection_dict["specific_neurons_L1"]
    specific_neurons_L2 = neuron_detection_dict["specific_neurons_L2"]
    non_activated_neurons_all = neuron_detection_dict["non_activated_neurons_all"]
    # print(activated_neurons_L1)
    # sys.exit()

    """ visualization """
    visualization_funcs.visualize_neurons_with_line_plot(
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
                                                        "tatoeba"
                                                        )

# if __name__ == "__main__":

    # for layer_idx, neurons in activated_neurons_L1:
    #     print(f"Layer {layer_idx}: Shared Neurons: {neurons}")
    # print(f'len_of_nonactivated_neurons_L2: {len(non_activated_neurons_L2)}, len_of_non_activated_neurons_L1: {len(non_activated_neurons_L1)}')
    # print(f'non_activated_neurons_all: {non_activated_neurons_all}')
