import itertools
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from baukit import Trace, TraceDict
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModel
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_out_llama3_self_att(model, prompt, device):
  model.eval() # swith the model to evaluation mode (deactivate dropout, batch normalization)
  num_layers = model.config.num_hidden_layers  # nums of layers of the model
  SELF_ATT_values = [f"model.layers.{i}.self_attn" for i in range(num_layers)]  # generate path to MLP layer(of LLaMA-3)

  with torch.no_grad():
      # trace MLP layers using TraceDict
      with TraceDict(model, SELF_ATT_values) as ret:
          output = model(prompt, output_hidden_states=True, output_attentions=True)  # run inference
      SELF_ATT_value = [ret[act_value].output for act_value in SELF_ATT_values]  # get outputs of self_att per layer
      return SELF_ATT_value

def get_outputs_self_att(model, input_ids):
  SELF_ATT_values = get_out_llama3_self_att(model, input_ids, model.device)  # Llamaのself-att直後の値を取得
  # SELF_ATT_values = [act for act in SELF_ATT_values]
  SELF_ATT_values = [act[0].cpu() for act in SELF_ATT_values] # act[0]: tuple(attention_output, attention_weights, cache) <- act[0](attention_output)のみが欲しいのでそれをcpu上に配置

  return SELF_ATT_values

def get_similarities_self_att(model, tokenizer, data) -> defaultdict(list):
  """
  get cosine similarities of self_att(or embeddings) block outputs of all the sentences in data (same semantics or non-same semantics pairs)

  block: "self_att" or "embeddings"

  return: defaultdict(list):
      {
        layer_idx: [] <- list of similarities per a pair
      }
  """
  similarities = defaultdict(list)
  for L1_text, L2_text in data:
    input_ids_L1 = tokenizer(L1_text, return_tensors="pt").input_ids.to(device)
    token_len_L1 = len(input_ids_L1[0])
    input_ids_L2 = tokenizer(L2_text, return_tensors="pt").input_ids.to(device)
    token_len_L2 = len(input_ids_L2[0])

    output_L1 = get_outputs_self_att(model, input_ids_L1)
    output_L2 = get_outputs_self_att(model, input_ids_L2)
    """
    shape of outputs: 32(layer_num) lengthのリスト。
    outputs[0]: 0層目のattentionを通った後のrepresentation
    """
    # attentionの出力を取得
    for layer_idx in range(32):
      """
      各レイヤーの最後のトークンに対応するattention_outputを取得（output_L1[layer_idx][0][token_len_L1-1]) + 2次元にreshape(2次元にreshapeしないとcos_simが測れないため。)
      """
      similarity = cosine_similarity(output_L1[layer_idx][0][token_len_L1-1].unsqueeze(0), output_L2[layer_idx][0][token_len_L2-1].unsqueeze(0))
      similarities[layer_idx].append(similarity[0][0]) # for instance, similarity=[[0.93852615]], so remove [[]] and extract similarity value only

  return similarities

# cosine similarity
def plot_hist(dict1: defaultdict(float), dict2: defaultdict(float), L2: str) -> None:
    # convert keys and values into list
    keys = list(dict1.keys())
    values1 = list(dict1.values())
    values2 = list(dict2.values())

    # plot hist
    plt.bar(keys, values1, alpha=1, label='same semantics')
    plt.bar(keys, values2, alpha=1, label='different semantics')

    plt.xlabel('Layer index')
    plt.ylabel('Cosine Similarity')
    plt.title(f'en_{L2}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/home/s2410121/proj_LA/measure_similarities/llama3/images/attention_outputs_sim/cos_sim/base/llama3_attention_outputs_base_sim_en_{L2}.png")
    plt.close()


if __name__ == "__main__":
  """ model configs """
  # LLaMA-3
  model_names = {
      # "base": "meta-llama/Meta-Llama-3-8B",
      "ja": "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
      # "de": "DiscoResearch/Llama3-German-8B", # ger
      "nl": "ReBatch/Llama-3-8B-dutch", # du
      "it": "DeepMount00/Llama-3-8b-Ita", # ita
      "ko": "beomi/Llama-3-KoEn-8B", # ko
  }

  for L2, model_name in model_names.items():
    L1 = "en" # L1 is fixed to english.

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B").to(device).eval()

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

    """
    baseとして、対訳関係のない1文ずつのペアを作成
    (L1(en)はhttps://huggingface.co/datasets/agentlans/high-quality-english-sentences,
    L2はtatoebaの該当データを使用)
    """
    random_data = []
    # L1(en)
    en_base_ds = load_dataset("agentlans/high-quality-english-sentences")
    random_data_en = en_base_ds["train"][:num_sentences]
    en_base_ds_idx = 0
    for item in dataset:
        random_data.append((random_data_en["text"][en_base_ds_idx], item["translation"][L2]))
        en_base_ds_idx += 1

    """ calc similarities """
    results_same_semantics = get_similarities_self_att(model, tokenizer, tatoeba_data)
    results_non_same_semantics = get_similarities_self_att(model, tokenizer, random_data)
    final_results_same_semantics = defaultdict(float)
    final_results_non_same_semantics = defaultdict(float)
    for layer_idx in range(32): # ３２層
        final_results_same_semantics[layer_idx] = np.array(results_same_semantics[layer_idx]).mean()
        final_results_non_same_semantics[layer_idx] = np.array(results_non_same_semantics[layer_idx]).mean()


    # delete some cache
    del model
    torch.cuda.empty_cache()

    """ plot """
    plot_hist(final_results_same_semantics, final_results_non_same_semantics, L2)

  print("visualization completed !")