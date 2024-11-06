""" neurons detection """

import sys

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch

from baukit import Trace, TraceDict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModel

from detect_act_neurons import *

""" models """
# LLaMA-3
model_names = {
    "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1",
    # "ger": "DiscoResearch/Llama3-German-8B",
    # "du": "ReBatch/Llama-3-8B-dutch",
    # "ita": "DeepMount00/Llama-3-8b-Ita",
    # "ko": "beomi/Llama-3-KoEn-8B",
}

L1 = "en" # L1 is fixed to english.

for L2, model_name in model_names.items():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    MODEL = 'llama3'

    """ dict for tracking activation count """
    act_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    """ dict for tracking non-activation count """
    non_act_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # {
    #     L1: {
    #         layer_idx: {
    #               neuron_idx: 32
    #           }
    #     }
    #     L2: ...
    #     ....
    # }

    """ Tatoeba translation corpus """
    tatoeba_data = [
        # ("이것은 테스트입니다.", "This is a test."), # ko
        # ("안녕하세요.", "Hello."),
        # ("안녕히 가세요.", "Goodbye."),
        # ("잘 지내세요?", "How are you?")
        ("これはテストです。", "This is a test."), # ja
        ("こんにちは。", "Hello."),
        ("さようなら。", "Goodbye."),
        ("お元気ですか？", "How are you?"),
        # ("Dies ist ein Test.", "This is a test."), # ger
        # ("Hallo.", "Hello."),
        # ("Auf Wiedersehen.", "Goodbye."),
        # ("Wie geht's?", "How are you?"),
        # ("Dit is een test.", "This is a test."), # du
        # ("Hallo.", "Hello."),
        # ("Vaarwel.", "Goodbye."),
        # ("Hoe gaat het?", "How are you?"),
        # ("Questo è un test.", "This is a test."), # ita
        # ("Ciao.", "Hello."),
        # ("Arrivederci.", "Goodbye."),
        # ("Come stai?", "How are you?"),
    ]
    # Dataset({
    #     features: ['id', 'translation'],
    #     num_rows: 208866
    # })
    # dataset = load_dataset("tatoeba", lang1=L1, lang2=L2, split="train")
    # tatoeba_data = [(item['translation'][L1], item['translation'][L2]) for item in dataset]
    # tatoeba_data_len = len(tatoeba_data)

    """ loop for each sentence of tatoeba """
    for L1, L2 in tatoeba_data:
        # L1 text
        input_ids_L1 = tokenizer(L1, return_tensors="pt").input_ids.to("cuda")
        mlp_activation_L1 = act_llama3(model, input_ids_L1)
        # L2 text
        input_ids_L2 = tokenizer(L2, return_tensors="pt").input_ids.to("cuda")
        mlp_activation_L2 = act_llama3(model, input_ids_L2)

        """ detect each type of neurons """
        # list for tracking neurons
        activated_neurons_L2 = []
        activated_neurons_L1 = []
        non_activated_neurons_L2 = []
        non_activated_neurons_L1 = []

        # aggregate each type of neurons per each layer
        for layer_idx in range(len(mlp_activation_L2)):

            # activated neurons for L1
            activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] > 0).cpu().numpy()  # convert to numpy array after move to CPU
            activated_neurons_L1.append((layer_idx, activated_neurons_L1_layer))

            # activated neurons for L2
            activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] > 0).cpu().numpy()
            activated_neurons_L2.append((layer_idx, activated_neurons_L2_layer))

            # non-activated neurons for L1
            non_activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] <= 0).cpu().numpy()
            non_activated_neurons_L1.append((layer_idx, non_activated_neurons_L1_layer))

            # non-activated neurons for L2
            non_activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] <= 0).cpu().numpy()
            non_activated_neurons_L2.append((layer_idx, non_activated_neurons_L2_layer))

        """ track activation/non-activation count of every neuron """
        # print(activated_neurons_L1)
        # print(len(activated_neurons_L1))
        # sys.exit()
        # for L1 activated_neurons
        for neuron_idx in activated_neurons_L1[1]:
            act_count[L1][layer_idx][neuron_idx] += 1
        # for L2 activated_neurons
        for neuron_idx in activated_neurons_L2[1]:
            act_count[L2][layer_idx][neuron_idx] += 1
        # for L1 non-activated_neurons
        for neuron_idx in non_activated_neurons_L1[1]:
            non_act_count[L1][layer_idx][neuron_idx] += 1
        for neuron_idx in non_activated_neurons_L1:
            non_act_count[L2][layer_idx][neuron_idx][1] += 1


    """ Check if activation count of each neuron is exceed THRESHOLD """
    # THRESHOLD = 0.7 * len(tatoeba_data)
    THRESHOLD = 0.7 * 4

    # list for tracking neurons
    activated_neurons_L1 = []
    activated_neurons_L2 = []
    non_activated_neurons_L1 = []
    non_activated_neurons_L2 = []

    # for activated neurons
    for lang, layer_idx in act_count.items():
        for layer_idx, neuron_idx in layer_idx.items():
            """ append to activated neurons if activation count is exceeded THRESHOLD """
            if lang == L1 and layer_idx[neuron_idx] >= int(THRESHOLD):
                activated_neurons_L1.append((layer_idx, neuron_idx))
            elif lang == L2 and layer_idx[neuron_idx] >= int(THRESHOLD):
                activated_neurons_L2.append(((layer_idx, neuron_idx)))

    # for non activated neurons
    for lang, layer_idx in non_act_count.items():
        for layer_idx, neuron_idx in layer_idx.items():
            """ append to activated neurons if activation count is exceeded THRESHOLD """
            if lang == L1 and layer_idx[neuron_idx] >= int(THRESHOLD):
                non_activated_neurons_L1.append((layer_idx, neuron_idx))
            elif lang == L2 and layer_idx[neuron_idx] >= int(THRESHOLD):
                non_activated_neurons_L2.append(((layer_idx, neuron_idx)))

    print(activated_neurons_L1, len(activated_neurons_L1))
    print(activated_neurons_L2, len(activated_neurons_L2))
    print(non_activated_neurons_L1, len(non_activated_neurons_L1))
    print(non_activated_neurons_L2, len(non_activated_neurons_L2))
    sys.exit()


if __name__ == "__main__":

    # for layer_idx, neurons in shared_neurons:
    #     print(f"Layer {layer_idx}: Shared Neurons: {neurons}")
    print(len(non_activated_neurons_L2), len(non_activated_neurons_L1))
    print(len(non_activated_neurons_all))
