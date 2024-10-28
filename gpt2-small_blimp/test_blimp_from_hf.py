""" huggingfaceからBLiMPを使えるかテスト """

import transformers

from datasets import get_dataset_config_names, load_dataset

configs = get_dataset_config_names("blimp") # configsのlistを取得
# print(configs)

configs_can_be_loaded = []
for config in configs:
    data = load_dataset("blimp", config)

