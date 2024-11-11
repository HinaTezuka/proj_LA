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
    token_len = len(inputs["input_ids"][0])

    # inference
    with torch.no_grad():
        model(**inputs)

    # remove hook_fn from MLP
    for hook in hooks:
        hook.remove()

    return activations, token_len

""" 最初からsetで管理 """
def track_neurons_with_text_data(model, model_name, tokenizer, data, active_THRESHOLD=0, non_active_THRESHOLD=0) -> dict:
    # Initialize lists for tracking neurons
    activated_neurons_L2 = []
    activated_neurons_L1 = []
    non_activated_neurons_L2 = []
    non_activated_neurons_L1 = []
    shared_neurons = []
    specific_neurons_L2 = []
    specific_neurons_L1 = []
    non_activated_neurons_all = []

    # Initialize sets for visualization
    num_layers = 32 if model_name == "llama" else 12
    activated_neurons_L2_vis = [set() for _ in range(num_layers)]
    activated_neurons_L1_vis = [set() for _ in range(num_layers)]
    non_activated_neurons_L2_vis = [set() for _ in range(num_layers)]
    non_activated_neurons_L1_vis = [set() for _ in range(num_layers)]
    shared_neurons_vis = [set() for _ in range(num_layers)]
    specific_neurons_L2_vis = [set() for _ in range(num_layers)]
    specific_neurons_L1_vis = [set() for _ in range(num_layers)]
    non_activated_neurons_all_vis = [set() for _ in range(num_layers)]

    # Track neurons with tatoeba
    for L1_text, L2_text in data:
        # L1 text
        # input_ids_L1 = tokenizer(L1_text, return_tensors="pt")
        mlp_activation_L1, token_len_L1 = get_activations(model, tokenizer, L1_text)
        # L2 text
        # input_ids_L2 = tokenizer(L2_text, return_tensors="pt")
        mlp_activation_L2, token_len_L2 = get_activations(model, tokenizer, L2_text)

        for layer_idx in range(12):
            # Activated neurons for L1 and L2
            activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] > active_THRESHOLD).cpu().numpy()
            # 最初の文頭トークンをカウントしない
            # activated_neurons_L1_layer = activated_neurons_L1_layer[activated_neurons_L1_layer[:, 1] != 0]
            # 最後のトークンだけ考慮する
            activated_neurons_L1_layer = activated_neurons_L1_layer[activated_neurons_L1_layer[:, 1] == token_len_L1 - 1]
            activated_neurons_L1.append((layer_idx, activated_neurons_L1_layer))
            activated_neurons_L1_vis[layer_idx].update(np.unique(activated_neurons_L1_layer[:, 2]))

            activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] > active_THRESHOLD).cpu().numpy()
            # activated_neurons_L2_layer = activated_neurons_L2_layer[activated_neurons_L2_layer[:, 1] != 0]
            # 最後のトークンだけ考慮する
            activated_neurons_L2_layer = activated_neurons_L2_layer[activated_neurons_L2_layer[:, 1] == token_len_L2 - 1]
            activated_neurons_L2.append((layer_idx, activated_neurons_L2_layer))
            activated_neurons_L2_vis[layer_idx].update(np.unique(activated_neurons_L2_layer[:, 2]))

            # Non-activated neurons for L1 and L2
            non_activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] <= non_active_THRESHOLD).cpu().numpy()
            # non_activated_neurons_L1_layer = non_activated_neurons_L1_layer[non_activated_neurons_L1_layer[:, 1] != 0]
            # 最後のトークンだけ考慮する
            non_activated_neurons_L1_layer = non_activated_neurons_L1_layer[non_activated_neurons_L1_layer[:, 1] == token_len_L1 - 1]
            non_activated_neurons_L1.append((layer_idx, non_activated_neurons_L1_layer))
            non_activated_neurons_L1_vis[layer_idx].update(np.unique(non_activated_neurons_L1_layer[:, 2]))

            non_activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] <= non_active_THRESHOLD).cpu().numpy()
            # non_activated_neurons_L2_layer = non_activated_neurons_L2_layer[non_activated_neurons_L2_layer[:, 1] != 0]
            # 最後のトークンだけ考慮する
            non_activated_neurons_L2_layer = non_activated_neurons_L2_layer[non_activated_neurons_L2_layer[:, 1] == token_len_L2 - 1]
            non_activated_neurons_L2.append((layer_idx, non_activated_neurons_L2_layer))
            non_activated_neurons_L2_vis[layer_idx].update(np.unique(non_activated_neurons_L2_layer[:, 2]))

            # Non-activated neurons for both L1 and L2
            non_activated_neurons_both_L1_L2_layer = np.intersect1d(non_activated_neurons_L1_layer[:, 2], non_activated_neurons_L2_layer[:, 2])
            non_activated_neurons_all.append((layer_idx, non_activated_neurons_both_L1_L2_layer))
            non_activated_neurons_all_vis[layer_idx].update(non_activated_neurons_both_L1_L2_layer)

            # Shared neurons
            shared_neurons_layer = np.intersect1d(activated_neurons_L1_layer[:, 2], activated_neurons_L2_layer[:, 2])
            shared_neurons.append((layer_idx, shared_neurons_layer))
            shared_neurons_vis[layer_idx].update(shared_neurons_layer)

            # Specific neurons
            specific_neurons_L1_layer = np.intersect1d(activated_neurons_L1_layer[:, 2], non_activated_neurons_L2_layer[:, 2])
            specific_neurons_L1.append((layer_idx, specific_neurons_L1_layer))
            specific_neurons_L1_vis[layer_idx].update(specific_neurons_L1_layer)

            specific_neurons_L2_layer = np.intersect1d(activated_neurons_L2_layer[:, 2], non_activated_neurons_L1_layer[:, 2])
            specific_neurons_L2.append((layer_idx, specific_neurons_L2_layer))
            specific_neurons_L2_vis[layer_idx].update(specific_neurons_L2_layer)

    # Create output dictionaries
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

    output_dict_vis = {
        "activated_neurons_L1": activated_neurons_L1_vis,
        "activated_neurons_L2": activated_neurons_L2_vis,
        "non_activated_neurons_L1": non_activated_neurons_L1_vis,
        "non_activated_neurons_L2": non_activated_neurons_L2_vis,
        "shared_neurons": shared_neurons_vis,
        "specific_neurons_L1": specific_neurons_L1_vis,
        "specific_neurons_L2": specific_neurons_L2_vis,
        "non_activated_neurons_all": non_activated_neurons_all_vis
    }

    return output_dict, output_dict_vis

# def track_neurons_with_text_data(model, tokenizer, data, active_THRESHOLD=0, non_active_THRESHOLD=0) -> dict: # data <- 通常はtatoeba
#    # list for tracking neurons
#     activated_neurons_L2 = []
#     activated_neurons_L1 = []
#     non_activated_neurons_L2 = []
#     non_activated_neurons_L1 = []
#     shared_neurons = []
#     specific_neurons_L2 = []
#     specific_neurons_L1 = []
#     non_activated_neurons_all = []

#     """ track neurons with tatoeba """
#     for L1_text, L2_text in data:
#         # L1 text
#         # input_ids_L1 = tokenizer(L1_text, return_tensors="pt")
#         mlp_activation_L1 = get_activations(model, tokenizer, L1_text)
#         # L2 text
#         # input_ids_L2 = tokenizer(L2_text, return_tensors="pt")
#         mlp_activation_L2 = get_activations(model, tokenizer, L2_text)

#         """ aggregate each type of neurons per each layer """
#         for layer_idx in range(len(mlp_activation_L2)):

#             """ activated neurons """
#             # activated neurons for L1
#             # torch.nonzero return index, not actual activation values
#             activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] > active_THRESHOLD).cpu().numpy()
#             # remove activation to 0-th token (for gpt2, it's <|begin_of_text|>)
#             activated_neurons_L1_layer = activated_neurons_L1_layer[activated_neurons_L1_layer[:, 1] != 0]
#             activated_neurons_L1.append((layer_idx, activated_neurons_L1_layer))
#             # activated neurons for L2
#             activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] > active_THRESHOLD).cpu().numpy()
#             # remove activation to 0-th token (for gpt2, it's <|begin_of_text|>)
#             activated_neurons_L2_layer = activated_neurons_L2_layer[activated_neurons_L2_layer[:, 1] != 0]
#             activated_neurons_L2.append((layer_idx, activated_neurons_L2_layer))

#             """ non-activated neurons """
#             # non-activated neurons for L1
#             non_activated_neurons_L1_layer = torch.nonzero(mlp_activation_L1[layer_idx] <= non_active_THRESHOLD).cpu().numpy()
#             non_activated_neurons_L1.append((layer_idx, non_activated_neurons_L1_layer))
#             # non-activated neurons for L2
#             non_activated_neurons_L2_layer = torch.nonzero(mlp_activation_L2[layer_idx] <= non_active_THRESHOLD).cpu().numpy()
#             non_activated_neurons_L2.append((layer_idx, non_activated_neurons_L2_layer))
#             # non-activated_neurons for both L1 and L2
#             non_activated_neurons_both_L1_L2_layer = np.intersect1d(non_activated_neurons_L1_layer, non_activated_neurons_L2_layer)
#             non_activated_neurons_all.append((layer_idx, non_activated_neurons_both_L1_L2_layer))

#             """ shared neurons """
#             # shared neurons for both L1 and L2
#             shared_neurons_layer = np.intersect1d(activated_neurons_L2_layer, activated_neurons_L1_layer)
#             shared_neurons.append((layer_idx, shared_neurons_layer))

#             """ specific neurons """
#             # specific neurons for L1
#             specific_neurons_L1_layer = np.intersect1d(activated_neurons_L1_layer, non_activated_neurons_L2_layer)
#             specific_neurons_L1_layer = np.intersect1d(specific_neurons_L1_layer, shared_neurons_layer)
#             # specific_neurons_L1_layer = np.intersect1d(specific_neurons_L1_layer, non_activated_neurons_both_L1_L2_layer)
#             specific_neurons_L1.append((layer_idx, specific_neurons_L1_layer))
#             # specific neurons for L2
#             specific_neurons_L2_layer = np.intersect1d(activated_neurons_L2_layer, non_activated_neurons_L1_layer)
#             specific_neurons_L2_layer = np.intersect1d(specific_neurons_L2_layer, shared_neurons_layer)
#             specific_neurons_L2.append((layer_idx, specific_neurons_L2_layer))

#     output_dict = {
#         "activated_neurons_L1": activated_neurons_L1,
#         "activated_neurons_L2": activated_neurons_L2,
#         "non_activated_neurons_L1": non_activated_neurons_L1,
#         "non_activated_neurons_L2": non_activated_neurons_L2,
#         "shared_neurons": shared_neurons,
#         "specific_neurons_L1": specific_neurons_L1,
#         "specific_neurons_L2": specific_neurons_L2,
#         "non_activated_neurons_all": non_activated_neurons_all
#     }

#     return output_dict
