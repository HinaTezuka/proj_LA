from functools import partial

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルとトークナイザーの初期化
model_name = "tokyotech-llm/Llama-3-Swallow-8B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPUが使用可能な場合はGPUを使用
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# MLPの出力サイズの取得
num_layers = model.config.num_hidden_layers
intermediate_size = model.config.intermediate_size  # MLP層の中間サイズを取得
layer_output_size = model.config.hidden_size  # 最終出力のサイズ

# 活性化の集計用テンソルの初期化
sum1 = torch.zeros(num_layers, layer_output_size).to(device)
sum2 = torch.zeros(num_layers, layer_output_size).to(device)
sum3 = torch.zeros(num_layers, layer_output_size).to(device)
sum4 = torch.zeros(num_layers, layer_output_size).to(device)
over_zero = torch.zeros(num_layers, layer_output_size, dtype=torch.int32).to(device)

# MLPのフォワードパスをフックする関数
def hook_fn(module, input, output, layer_idx):
    activation = output.float()
    sum1[layer_idx, :] += activation.sum(dim=(0, 1))
    sum2[layer_idx, :] += activation.pow(2).sum(dim=(0, 1))
    sum3[layer_idx, :] += activation.pow(3).sum(dim=(0, 1))
    sum4[layer_idx, :] += activation.pow(4).sum(dim=(0, 1))
    over_zero[layer_idx, :] += (activation > 0).sum(dim=(0, 1))

# 各MLP層にフックを設定
# for i in range(num_layers):
#     mlp_layer = model.model.layers[i].mlp  # Llamaの場合、MLP層を参照
#     mlp_layer.register_forward_hook(lambda m, inp, out: hook_fn(m, inp, out, i))

# 各MLP層にフックを設定
for i in range(num_layers):
    mlp_layer = model.model.layers[i].mlp  # Llamaの場合、MLP層を参照
    # partialを使ってlayer_idxを固定したフックを作成
    mlp_layer.register_forward_hook(partial(hook_fn, layer_idx=i))

# 入力テキストの準備
input_text = "こんにちは。今日の新聞でした。"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# モデルの生成を実行
with torch.no_grad():
    _ = model.generate(input_ids=input_ids, max_length=50)

# 結果の出力
output = {
    "sum1": sum1.cpu(),
    "sum2": sum2.cpu(),
    "sum3": sum3.cpu(),
    "sum4": sum4.cpu(),
    "over_zero": over_zero.cpu(),
}
for k, v in output.items():
    print(f'shape of {k}: {v.shape}')
print(output['over_zero'])
# 結果の保存
torch.save(output, 'activation.ja.llama3')

print("活性化の統計を保存しました。")

# # n を定義 (活性化を記録するための入力長)
# n = input_ids.size(1)  # 入力トークンの数

# 結果の出力
# output = {
#     "n": n,
#     "sum1": sum1.cpu(),
#     "sum2": sum2.cpu(),
#     "sum3": sum3.cpu(),
#     "sum4": sum4.cpu(),
#     "over_zero": over_zero.cpu(),
# }

# # for k, v in output.items():
# #     print(f'shape of {k}: {v.shape}')
# print(output['over_zero'])

# # 結果の保存
# torch.save(output, 'activation.ja.llama3')

# print("活性化の統計を保存しました。")

# 発火しているニューロン（0より多く発火したもの）を確認
layer_idx = 0  # 対象の層
threshold = 0  # 発火回数の閾値（1以上で発火と見なす）

firing_neurons = (output["over_zero"][layer_idx] > threshold).nonzero(as_tuple=True)[0]
print(f"Layer {layer_idx} - Firing neurons (above threshold {threshold}):", firing_neurons.tolist())
