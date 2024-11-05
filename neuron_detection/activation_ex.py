"""
getting activated neurons inside models

refer to: https://github.com/RUCAIBox/Language-Specific-Neurons/blob/main/activation.py
focus on: MLP layers
"""
import sys
from types import MethodType

import torch
import transformers

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

""" load models """
model_name = "tokyotech-llm/Llama-3-Swallow-8B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
# flag
is_llama = "llama" in model_name.lower()

""" track neuron-activation """
# configs
max_length = model.config.max_position_embeddings
num_layers = model.config.num_hidden_layers
# intermediate_size = model.config.hidden_size * 4 if not is_llama else model.config.intermediate_size
intermediate_size = model.config.intermediate_size if is_llama else model.config.hidden_size * 4

# initialize tensors to track activation
sum1 = torch.zeros(num_layers, intermediate_size // 2).to('cuda')
print(sum1.shape)
sum2 = torch.zeros(num_layers, intermediate_size // 2).to('cuda')
sum3 = torch.zeros(num_layers, intermediate_size // 2).to('cuda')
sum4 = torch.zeros(num_layers, intermediate_size // 2).to('cuda')
over_zero = torch.zeros(num_layers, intermediate_size // 2, dtype=torch.int32).to('cuda')

# forward method for LLaMA-3
def llama_forward(self, x, idx):
    # 入力xに対してゲートアップ投影を適用
    print(self)
    gate_up = self.gate_proj(x)
    # gate_upの最後の次元のサイズを取得、iに追加
    i = gate_up.size(-1)
    print(f"gate_up shape: {gate_up.shape}")
    sys.exit()
    #　gate_up の最初の半分の部分に対して、SiLUを適用/非線形変換（最後の次元から最初の半分の要素をスライスで選択）
    # MLP（多層パーセプトロン）では、通常、入力を2つの部分に分けて、一部は活性化を適用し、もう一部はスケーリングに使われることが多い
    gate_up[:, :, : i // 2] = torch.nn.SiLU()(gate_up[:, :, : i // 2])
    # SiLU活性化関数を通過した部分をactivationに格納(activationは後で使用するため、浮動小数点型に変換)
    activation = gate_up[:, :, : i // 2].float()
    """
    activationの値をすべて足し合わせて、sum1のidx層に加算。これにより、この層での活性化の合計が追跡される
    sum1-4は、それぞれニューロンの活性具合を強調させるために、指標として異なる乗数を計算
    """
    # activationの形状を確認
    print(f"sums shape: {sum1.shape}")
    print("Activation shape:", activation.shape)
    sum1[idx, :] += activation.sum(dim=(0, 1))
    # activationの値を二乗したものを合計し、sum2に加える。これは、活性化の二乗の合計を追跡するため
    sum2[idx, :] += activation.pow(2).sum(dim=(0, 1))
    sum3[idx, :] += activation.pow(3).sum(dim=(0, 1))
    sum4[idx, :] += activation.pow(4).sum(dim=(0, 1))
    # ０より大きいactivationsを取得
    over_zero[idx, :] += (activation > 0).sum(dim=(0, 1))
    print(gate_up[:, :, : i // 2].shape, gate_up[:, :, i // 2 :].shape)
    # 出力の生成
    x = gate_up[:, :, : i // 2] * gate_up[:, :, i // 2 :]
    # x = gate_up[:, :, : i] * gate_up[:, :, i :]
    print(x.shape)
    print(f'over_zero: {over_zero}')
    # sys.exit()
    x, _ = self.down_proj(x)
    # print(x.shape())
    # sys.exit()
    return x

# factory関数で層の`forward`メソッドを上書き
def factory(idx):
    if is_llama:
        return lambda self, x: llama_forward(self, x, idx)

# 各層のMLPのforwardメソッドを上書き
for i in range(num_layers):
    if is_llama:
        obj = model.model.layers[i].mlp

    obj.forward = MethodType(factory(i), obj)


tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "サンプルテキスト"  # テスト用
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

# implement inference
output = model.generate(input_ids=input_ids, max_length=max_length)

# 結果の保存
output_dict = {
    "sum1": sum1.to("cpu"),
    "sum2": sum2.to("cpu"),
    "sum3": sum3.to("cpu"),
    "sum4": sum4.to("cpu"),
    "over_zero": over_zero.to("cpu"),
}
# torch.save(output_dict, f"activation_data.pth")
