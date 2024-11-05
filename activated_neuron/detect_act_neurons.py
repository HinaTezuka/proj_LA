"""
detect activated neurons.
some codes are citation from: https://github.com/weixuan-wang123/multilingual-neurons/blob/main/neuron-behavior.ipynb
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from baukit import Trace, TraceDict
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModel
from datasets import load_dataset

model_name = "tokyotech-llm/Llama-3-Swallow-8B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
MODEL = 'llama3'

def get_out_llama3(model, prompt, device, index):
    model.eval() # swith the model to evaluation mode (deactivate dropout, batch normalization)
    num_layers = model.config.num_hidden_layers  # nums of layers of the model
    MLP_act = [f"model.layers.{i}.mlp" for i in range(num_layers)]  # generate path to MLP layer(of LLaMA-3)

    with torch.no_grad():
        # trace MLP layers using TraceDict
        with TraceDict(model, MLP_act) as ret:
            output = model(prompt, output_hidden_states=True, output_attentions=True)  # モデルを実行
        MLP_act_value = [ret[act_value].output for act_value in MLP_act]  # 各MLP層の出力を取得
        return MLP_act_value

def act_llama3(input_ids):
    mlp_act = get_out_llama3(model, input_ids, model.device, -1)  # LlamaのMLP活性化を取得
    mlp_act_tensors = [act.to("cpu") for act in mlp_act] # Numpy配列はCPUでしか動かないので、各テンソルをCPU上へ移動
    # mlp_act = np.array(mlp_act)  # convert to numpy array
    mlp_act_np = [act.detach().numpy() for act in mlp_act_tensors]
    return mlp_act

# if 'bloom' in MODEL:
#     LAYERS = model.config.n_layer
#     Neuron_num = 16384
# print(model)


# input_text = "test test test."  # テスト用のテキスト
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
# mlp_activation = act_llama3(input_ids)  # 活性化値を取得
# print("MLP activation values:", mlp_activation)
# print(mlp_activation[0].shape) # torch.Size([1, 9, 4096])　← batch_size=1, units=9, output of each unit=4096

""" ニューロンの重なりを判定 """
# Tatoebaデータを読み込む（ここでは例として固定のデータを使用）
# tatoeba_data = [
#     ("これはテストです。", "This is a test."),
#     ("こんにちは。", "Hello."),
#     ("さようなら。", "Goodbye."),
#     ("お元気ですか？", "How are you?"),
# ]

# Tatoebaデータセットのロード
# Dataset({
#     features: ['id', 'translation'],
#     num_rows: 208866
# })
dataset = load_dataset("tatoeba", lang1="en", lang2="ja", split="train")
print(dataset.features['translation'])
# データセットから英語-日本語ペアを取得
tatoeba_data = [(item['translation']['en'], item['translation']['ja']) for item in dataset]

# 各言語に対する活性化ニューロンを追跡するリスト
activated_neurons_japanese = []
activated_neurons_english = []
shared_neurons = []
specific_neurons_japanese = []
specific_neurons_english = []

for japanese, english in tatoeba_data:
    # 日本語の入力
    input_ids_japanese = tokenizer(japanese, return_tensors="pt").input_ids.to("cuda")
    mlp_activation_japanese = act_llama3(input_ids_japanese)

    # 英語の入力
    input_ids_english = tokenizer(english, return_tensors="pt").input_ids.to("cuda")
    mlp_activation_english = act_llama3(input_ids_english)

    # 各層の活性化ニューロンを集計
    for layer_idx in range(len(mlp_activation_japanese)):
        # 日本語の入力から活性化ニューロンを取得
        activated_neurons_japanese_layer = torch.nonzero(mlp_activation_japanese[layer_idx] > 0).cpu().numpy()  # CPUに移動してからNumPyに変換
        activated_neurons_japanese.append((layer_idx, activated_neurons_japanese_layer))

        # 英語の入力から活性化ニューロンを取得
        activated_neurons_english_layer = torch.nonzero(mlp_activation_english[layer_idx] > 0).cpu().numpy()  # CPUに移動してからNumPyに変換
        activated_neurons_english.append((layer_idx, activated_neurons_english_layer))

        # 両方の言語に反応するニューロンのインデックスを取得
        shared_neurons_layer = np.intersect1d(activated_neurons_japanese_layer, activated_neurons_english_layer)
        shared_neurons.append((layer_idx, shared_neurons_layer))

        # 特異的ニューロンを確認
        specific_neurons_japanese_layer = activated_neurons_japanese_layer[~np.isin(activated_neurons_japanese_layer, shared_neurons_layer)]
        specific_neurons_japanese.append((layer_idx, specific_neurons_japanese_layer))

        specific_neurons_english_layer = activated_neurons_english_layer[~np.isin(activated_neurons_english_layer, shared_neurons_layer)]
        specific_neurons_english.append((layer_idx, specific_neurons_english_layer))

# 統計情報の表示
# print(len(activated_neurons_japanese))
print("Japanese activated neurons:", activated_neurons_japanese)
# print("English activated neurons:", activated_neurons_english)
# print("Shared neurons between Japanese and English:", shared_neurons)
# print("Specific neurons activated only by Japanese:", specific_neurons_japanese)
# print("Specific neurons activated only by English:", specific_neurons_english)
# 各層ごとに共有ニューロンのインデックスを表示
for layer_idx, neurons in shared_neurons:
    print(f"Layer {layer_idx} shared neurons: {neurons}")

# 層の数
num_layers = 32

# 各層のニューロンの活性化を集計するためのリスト
japanese_counts = [0] * num_layers
english_counts = [0] * num_layers
shared_counts = [0] * num_layers
specific_japanese_counts = [0] * num_layers
specific_english_counts = [0] * num_layers

# 各層におけるニューロンの活性化数をカウント
for layer_idx in range(num_layers):
    japanese_counts[layer_idx] = len(activated_neurons_japanese[layer_idx][1])  # 日本語のニューロン数
    english_counts[layer_idx] = len(activated_neurons_english[layer_idx][1])    # 英語のニューロン数
    shared_counts[layer_idx] = len(shared_neurons[layer_idx][1])                # 共有ニューロン数
    specific_japanese_counts[layer_idx] = len(specific_neurons_japanese[layer_idx][1])  # 日本語特有のニューロン数
    specific_english_counts[layer_idx] = len(specific_neurons_english[layer_idx][1])    # 英語特有のニューロン数

# プロット
plt.figure(figsize=(15, 10))
plt.plot(range(num_layers), japanese_counts, label='Japanese Activated Neurons', marker='o')
plt.plot(range(num_layers), english_counts, label='English Activated Neurons', marker='o')
plt.plot(range(num_layers), shared_counts, label='Shared Neurons', marker='o')
plt.plot(range(num_layers), specific_japanese_counts, label='Specific to Japanese', marker='o')
plt.plot(range(num_layers), specific_english_counts, label='Specific to English', marker='o')

plt.title('Neuron Activation Counts per Layer')
plt.xlabel('Layer Index')
plt.ylabel('Number of Activated Neurons')
plt.xticks(range(num_layers))
plt.legend()
plt.grid()
# plt.show()

plt.savefig('/home/s2410121/proj_LA/activated_neuron')
plt.close()
