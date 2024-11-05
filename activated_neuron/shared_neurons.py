""" ニューロンの重なりを判定 """
from detect_act_neurons import *

# Tatoeba data for Test
tatoeba_data = [
    ("これはテストです。", "This is a test."),
    ("こんにちは。", "Hello."),
    ("さようなら。", "Goodbye."),
    ("お元気ですか？", "How are you?"),
]

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
    # print(len(mlp_activation_japanese)) # 32

    # 英語の入力
    input_ids_english = tokenizer(english, return_tensors="pt").input_ids.to("cuda")
    mlp_activation_english = act_llama3(input_ids_english)

    # 各層の活性化ニューロンを集計
    for layer_idx in range(len(mlp_activation_japanese)):
        # 日本語の入力から活性化ニューロンを取得
        activated_neurons_japanese_layer = torch.nonzero(mlp_activation_japanese[layer_idx] > 0).cpu().numpy()  # CPUに移動してからNumPyに変換
        activated_neurons_japanese.append((layer_idx, activated_neurons_japanese_layer))
        print(activated_neurons_japanese)

        # 英語の入力から活性化ニューロンを取得
        activated_neurons_english_layer = torch.nonzero(mlp_activation_english[layer_idx] > 0).cpu().numpy()  # CPUに移動してからNumPyに変換
        activated_neurons_english.append((layer_idx, activated_neurons_english_layer))

        # 両方の言語に反応するニューロンのインデックスを取得
        shared_neurons_layer = np.intersect1d(activated_neurons_japanese_layer, activated_neurons_english_layer)
        shared_neurons.append((layer_idx, shared_neurons_layer))

        # specific neurons (intersections of )
        specific_neurons_japanese_layer = activated_neurons_japanese_layer[~np.isin(activated_neurons_japanese_layer, shared_neurons_layer)]
        specific_neurons_japanese.append((layer_idx, specific_neurons_japanese_layer))

        specific_neurons_english_layer = activated_neurons_english_layer[~np.isin(activated_neurons_english_layer, shared_neurons_layer)]
        specific_neurons_english.append((layer_idx, specific_neurons_english_layer))

for layer_idx, neurons in shared_neurons:
    print(f"Layer {layer_idx}: Shared Neurons: {neurons}")
