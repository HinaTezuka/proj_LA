""" visualize magnitude of changes of params with heatmap """
import os
import sys
sys.path.append("/home/s2410121/proj_LA/measure_similarities/gpt2/gpt2_measure_similarity.py")
import re

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from gpt2_measure_similarity import *

# Define architectures and layer indices
param_changes_per_arc = {
                            'attention': {},
                            'mlp': {},
                            'ln': {},
                        }

# get changes of weight per each langs <- only for abs changes of params(not for cos_sim and biases)
def get_param_changes(weight_changes, langs: str, param_changes_per_arc: dict) -> dict:
    for k, v in weight_changes[langs].items():
        # h.[].などの層のidx名を消す
        modified_key = re.sub(r'h\.\d+\.', '', k)
        if 'attn' in k and 'weight' in k:
            if modified_key not in param_changes_per_arc['attention']:
                param_changes_per_arc['attention'][modified_key] = []
                param_changes_per_arc['attention'][modified_key].append(v['mean_absolute_difference'])
            else:
                param_changes_per_arc['attention'][modified_key].append(v['mean_absolute_difference'])
        elif 'mlp' in k and 'weight' in k:
            if modified_key not in param_changes_per_arc['mlp']:
                param_changes_per_arc['mlp'][modified_key] = []
                param_changes_per_arc['mlp'][modified_key].append(v['mean_absolute_difference'])
            else:
                param_changes_per_arc['mlp'][modified_key].append(v['mean_absolute_difference'])
        # elif 'ln' in k and 'weight' in k:
        #     if modified_key not in param_changes_per_arc['ln']:
        #         param_changes_per_arc['ln'][modified_key] = []
        #         param_changes_per_arc['ln'][modified_key].append(v['mean_absolute_difference'])
        #     else:
        #         param_changes_per_arc['ln'][modified_key].append(v['mean_absolute_difference'])

    return param_changes_per_arc

# Plotting function
def plot_heatmap(param_changes_per_arc, title="heatmap_gpt2", lang_pair="en_ja") -> None:
    data = defaultdict(list)
    for v in param_changes_per_arc.values():
        for param_name in v.keys():
            data[param_name] = v[param_name]
    # layer_idx (vertical)
    layer_idx = list(range(len(next(iter(data.values())))))
    # layer_idx = list(range(12))
    layer_idx = layer_idx[::-1]
    # df creation
    df = pd.DataFrame(data)
    # plot
    plt.figure(figsize=(20, 10))
    sns.heatmap(df, annot=True, cmap='Greens', vmin=0.05, vmax=0.3, cbar_kws={'label': 'changes of weights'})
    plt.title(f'{title}{lang_pair}')
    # plt.xlabel('param_names')
    plt.ylabel('layer_idx')
    plt.savefig(f'/home/s2410121/proj_LA/measure_similarities/gpt2/images/heat_maps/{title}{lang_pair}.png')
    plt.close()


if __name__ == "__main__":
    """ gpt2 """
    # en_ja
    param_changes_per_arc = get_param_changes(weight_changes_gpt2, 'en_ja', param_changes_per_arc)
    # print(param_changes_per_arc)
    plot_heatmap(param_changes_per_arc, "magnitude_changes_of_weights_", "en_ja")

    # en_du
    param_changes_per_arc = get_param_changes(weight_changes_gpt2, 'en_du', param_changes_per_arc)
    plot_heatmap(param_changes_per_arc, "magnitude_changes_of_weights_", "en_du")

    # en_ger
    param_changes_per_arc = get_param_changes(weight_changes_gpt2, 'en_ger', param_changes_per_arc)
    plot_heatmap(param_changes_per_arc, "magnitude_changes_of_weights_", "en_ger")

    # en_ita
    param_changes_per_arc = get_param_changes(weight_changes_gpt2, 'en_ita', param_changes_per_arc)
    plot_heatmap(param_changes_per_arc, "magnitude_changes_of_weights_", "en_ita")

    # en_fre
    param_changes_per_arc = get_param_changes(weight_changes_gpt2, 'en_fre', param_changes_per_arc)
    plot_heatmap(param_changes_per_arc, "magnitude_changes_of_weights_", "en_fre")\

    # en_ko
    param_changes_per_arc = get_param_changes(weight_changes_gpt2, 'en_ko', param_changes_per_arc)
    plot_heatmap(param_changes_per_arc, "magnitude_changes_of_weights_", "en_ko")

    """ llama3 """

