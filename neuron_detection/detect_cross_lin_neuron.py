"""
cross-lingualに発火しているニューロンの特定

"""

from transformers import GPT2Tokenizer, GPT2Model, AutoModel, AutoTokenizer
import torch
from datasets import load_dataset

# モデルのロード
model_name = "rinna/japanese-gpt2-small"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 特定の層のニューロンの活性化を取得
def get_neuron_activations(input_ids, layer_index):
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer_index]
        print(hidden_states)
    # print(hidden_states)
    return hidden_states

# 例: 入力文をトークン化し、3層目のニューロンの活性化を取得
input_text = "これはテストです。"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
activations = get_neuron_activations(input_ids, 2)  # 3層目の活性化を取得
# print(activations)

def check_cross_lingual_activation(en_input_ids, ja_input_ids, layer_index, threshold=0.5):
    # ニューロンの活性化を取得
    en_activations = get_neuron_activations(en_input_ids, layer_index)
    ja_activations = get_neuron_activations(ja_input_ids, layer_index)

    # 活性化のサイズを揃える
    min_lenth = min(en_activations.shape[1], ja_activations.shape[1])
    en_activations = en_activations[:, :min_lenth]
    ja_activations = ja_activations[:, :min_lenth]

    # 活性化の比較
    common_active_neurons = torch.where(
        (en_activations > threshold) & (ja_activations > threshold)
    )[1]

    return common_active_neurons

# 例: 英語と日本語の文ペアでチェック
en_text = "This is a test."
ja_text = "これはテストです。"
en_input_ids = tokenizer.encode(en_text, return_tensors='pt')
ja_input_ids = tokenizer.encode(ja_text, return_tensors='pt')

common_neurons = check_cross_lingual_activation(en_input_ids, ja_input_ids, 2)
print(common_neurons)
