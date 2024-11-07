"""
detect activated neurons(for attention).
some codes are a citation from: https://github.com/weixuan-wang123/multilingual-neurons/blob/main/neuron-behavior.ipynb

About GPT2:
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(32000, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2SdpaAttention(
          (c_attn): Conv1D(nf=2304, nx=768)
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=32000, bias=False)
)
"""
import sys
sys.path.append("/home/s2410121/proj_LA/activated_neuron/attention")

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModel, GPT2Tokenizer
from datasets import load_dataset

def get_activations(model, tokenizer, text):
    # switch model to eval mode
    model.eval()
    # list for track activation values
    activations = []

    # define hook_fn to track MLP neurons
    def hook_fn(module, input, output):
        activations.append(output)

    # add hook_fn to each layer of MLP
    hooks = []
    for i in range(len(model.transformer.h)):  # GPT-2の全層に対して
        hook = model.transformer.h[i].mlp.act.register_forward_hook(hook_fn)
        hooks.append(hook)

    # tokenize
    inputs = tokenizer(text, return_tensors="pt")
    # inference
    with torch.no_grad():
        model(**inputs)

    # remove hook_fn from MLP
    for hook in hooks:
        hook.remove()

    return activations

def track_neurons_with_text_data(model, tokenizer, data, active_THRESHOLD=0, non_active_THRESHOLD=0) -> dict: # data <- 通常はtatoeba
   # list for tracking neurons
    activated_neurons_L2 = []
    activated_neurons_L1 = []
    non_activated_neurons_L2 = []
    non_activated_neurons_L1 = []
    shared_neurons = []
    specific_neurons_L2 = []
    specific_neurons_L1 = []
    non_activated_neurons_all = []

    """ track neurons with tatoeba """
    for L1_text, L2_text in data:
        # L1 text
        # input_ids_L1 = tokenizer(L1_text, return_tensors="pt")
        mlp_activation_L1 = get_activations(model, tokenizer, L1_text)
        # L2 text
        # input_ids_L2 = tokenizer(L2_text, return_tensors="pt")
        mlp_activation_L2 = get_activations(model, tokenizer, L2_text)

        """ aggregate each type of neurons per each layer """
        for layer_idx in range(len(mlp_activation_L2)):

            """ activated neurons """
            # activated neurons for L1
            activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] > active_THRESHOLD).cpu().numpy()
            # remove activation to 0-th token (for gpt2, it's <|begin_of_text|>)
            activated_neurons_L1_layer = activated_neurons_L1_layer[activated_neurons_L1_layer[:, 1] != 0]
            activated_neurons_L1.append((layer_idx, activated_neurons_L1_layer))
            # activated neurons for L2
            activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] > active_THRESHOLD).cpu().numpy()
            # remove activation to 0-th token (for gpt2, it's <|begin_of_text|>)
            activated_neurons_L2_layer = activated_neurons_L2_layer[activated_neurons_L2_layer[:, 1] != 0]
            activated_neurons_L2.append((layer_idx, activated_neurons_L2_layer))

            """ non-activated neurons """
            # non-activated neurons for L1
            non_activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] <= non_active_THRESHOLD).cpu().numpy()
            non_activated_neurons_L1.append((layer_idx, non_activated_neurons_L1_layer))
            # non-activated neurons for L2
            non_activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] <= non_active_THRESHOLD).cpu().numpy()
            non_activated_neurons_L2.append((layer_idx, non_activated_neurons_L2_layer))
            # non-activated_neurons for both L1 and L2
            non_activated_neurons_both_L1_L2_layer = np.intersect1d(non_activated_neurons_L1_layer, non_activated_neurons_L2_layer)
            non_activated_neurons_all.append((layer_idx, non_activated_neurons_both_L1_L2_layer))

            """ shared neurons """
            # shared neurons for both L1 and L2
            shared_neurons_layer = np.intersect1d(activated_neurons_L2_layer, activated_neurons_L1_layer)
            shared_neurons.append((layer_idx, shared_neurons_layer))

            """ specific neurons """
            # specific neurons for L1
            specific_neurons_L1_layer = np.intersect1d(activated_neurons_L1_layer, non_activated_neurons_L2_layer)
            specific_neurons_L1_layer = np.intersect1d(specific_neurons_L1_layer, shared_neurons_layer)
            # specific_neurons_L1_layer = np.intersect1d(specific_neurons_L1_layer, non_activated_neurons_both_L1_L2_layer)
            specific_neurons_L1.append((layer_idx, specific_neurons_L1_layer))
            # specific neurons for L2
            specific_neurons_L2_layer = np.intersect1d(activated_neurons_L2_layer, non_activated_neurons_L1_layer)
            specific_neurons_L2_layer = np.intersect1d(specific_neurons_L2_layer, shared_neurons_layer)
            specific_neurons_L2.append((layer_idx, specific_neurons_L2_layer))

    output_dict = {
        "activated_neurons_L1": activated_neurons_L1,
        "activated_neurons_L2": activated_neurons_L2,
        "non_activated_neurons_L1": non_activated_neurons_L1,
        "non_activated_neurons_L2": non_activated_neurons_L2,
        "shared_neurons": shared_neurons,
        "specific_neurons_L1": specific_neurons_L1,
        "specific_neurons_L2": specific_neurons_L2,
        "non_activated_neurons_all": non_activated_neurons_all
    }

    return output_dict

if __name__ == "__main__":
    neuron_detection_dict = track_neurons_with_text_data(model, tokenizer, tatoeba_data)

    activated_neurons_L1 = neuron_detection_dict["activated_neurons_L1"]
    activated_neurons_L2 = neuron_detection_dict["activated_neurons_L2"]
    non_activated_neurons_L1 = neuron_detection_dict["non_activated_neurons_L1"]
    non_activated_neurons_L2 = neuron_detection_dict["non_activated_neurons_L2"]
    shared_neurons = neuron_detection_dict["shared_neurons"]
    specific_neurons_L1 = neuron_detection_dict["specific_neurons_L1"]
    specific_neurons_L2 = neuron_detection_dict["specific_neurons_L2"]
    non_activated_neurons_all = neuron_detection_dict["non_activated_neurons_all"]

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

    """ visualization """
    visualize_neurons_with_line_plot(
                                        "en",
                                        "ja",
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
