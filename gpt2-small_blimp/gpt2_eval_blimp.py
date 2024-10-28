# from transformers import AutoModelForCausalLM, AutoTokenizer
# from collections import defaultdict
# from datasets import load_dataset
# import torch

# # 使用するモデルとトークナイザーのロード
# model_name = "gpt2"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # すべてのBLiMP項目のリスト
# configs = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive',
#            'animate_subject_trans', 'causative', 'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch',
#            'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2',
#            'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2',
#            'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1',
#            'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2',
#            'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2', 'existential_there_subject_raising',
#            'expletive_it_object_raising', 'inchoative', 'intransitive', 'irregular_past_participle_adjectives', 'irregular_past_participle_verbs',
#            'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question',
#            'left_branch_island_simple_question', 'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present',
#            'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2', 'principle_A_domain_1',
#            'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction', 'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 'sentential_subject_island',
#             'superlative_quantifiers_1', 'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island',
#             'wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap',
#             'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance'
#           ]

# # 評価関数
# def evaluate_sentence_pair(sentence1, sentence2):
#     inputs1 = tokenizer(sentence1, return_tensors="pt")
#     inputs2 = tokenizer(sentence2, return_tensors="pt")

#     with torch.no_grad():
#         outputs1 = model(**inputs1)
#         outputs2 = model(**inputs2)

#     # 各トークンの対数確率を合計してスコアを計算
#     score1 = outputs1.logits.log_softmax(dim=-1)[..., inputs1.input_ids[0]].sum()
#     score2 = outputs2.logits.log_softmax(dim=-1)[..., inputs2.input_ids[0]].sum()

#     return score1, score2

# # 各タスクの結果を保存する辞書
# task_results = defaultdict(lambda: {"correct": 0, "total": 0})

# # 各タスクを順に評価
# for config in configs:
#     blimp = load_dataset("blimp", config)  # 個別のタスクをロード
#     print(blimp)
#     for example in blimp["train"]:
#         sentence1 = example["sentence_good"]
#         sentence2 = example["sentence_bad"]
#         score1, score2 = evaluate_sentence_pair(sentence1, sentence2)

#         if score1 > score2:
#             task_results[config]["correct"] += 1
#         task_results[config]["total"] += 1

# # 各タスクごとの精度を計算して出力
# for task_name, counts in task_results.items():
#     accuracy = counts["correct"] / counts["total"]
#     print(f"{task_name} の精度: {accuracy * 100:.2f}%")


from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from datasets import load_dataset
import torch
import pandas as pd

# 使用するモデル名のリスト
model_names = [
                # gpt2-small
                "gpt2", # base model: original gpt2(small) model : en
                "rinna/japanese-gpt2-small", # ja
                "GroNLP/gpt2-small-dutch", # du
                "dbmdz/german-gpt2", # ger
                # "GroNLP/gpt2-small-italian", # ita
                # "dbddv01/gpt2-french-small", # fre
                # "skt/kogpt2-base-v2", # ko
                # "datificate/gpt2-small-spanish", # spa
                # llama3-8b
                # "meta-llama/Meta-Llama-3-8B", # en
                # "tokyotech-llm/Llama-3-Swallow-8B-v0.1", # ja
                # "DiscoResearch/Llama3-German-8B", # ger
                # "DeepMount00/Llama-3-8b-Ita", # ita
                # "beomi/Llama-3-KoEn-8B", # ko
              ]

# BLiMPの評価項目リスト
# configs = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive',
#            'animate_subject_trans', 'causative', 'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch',
#            'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2',
#            'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2',
#            'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1',
#            'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2',
#            'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2', 'existential_there_subject_raising',
#            'expletive_it_object_raising', 'inchoative', 'intransitive', 'irregular_past_participle_adjectives', 'irregular_past_participle_verbs',
#            'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question',
#            'left_branch_island_simple_question', 'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present',
#            'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2', 'principle_A_domain_1',
#            'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction', 'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 'sentential_subject_island',
#             'superlative_quantifiers_1', 'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island',
#             'wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap',
#             'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance'
#           ]
configs = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive', 'animate_subject_trans', 'causative', 'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch', 'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2', 'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2', 'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive', 'irregular_past_participle_adjectives', 'irregular_past_participle_verbs', 'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question', 'left_branch_island_simple_question', 'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1', 'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance']
# configs = ['adjunct_island', 'anaphor_gender_agreement']
# 評価関数
def evaluate_sentence_pair(model, tokenizer, sentence1, sentence2):
    inputs1 = tokenizer(sentence1, return_tensors="pt")
    inputs2 = tokenizer(sentence2, return_tensors="pt")

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    """ モデルがそれぞれの文を生成する確率 """
    score1 = outputs1.logits.log_softmax(dim=-1)[..., inputs1.input_ids[0]].sum()
    score2 = outputs2.logits.log_softmax(dim=-1)[..., inputs2.input_ids[0]].sum()

    return score1, score2

# データを保存するリスト
results = []

# 各モデルについてBLiMPのタスクを評価
for model_name in model_names:
    # モデルとトークナイザーをロード
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 各評価項目ごとに評価
    for config in configs:
        blimp = load_dataset("blimp", config)
        correct = 0
        total = 0

        for example in blimp["train"]:
            sentence1 = example["sentence_good"]
            sentence2 = example["sentence_bad"]
            score1, score2 = evaluate_sentence_pair(model, tokenizer, sentence1, sentence2)

            if score1 > score2:
                correct += 1
            total += 1

        # 精度を計算して結果を保存
        accuracy = correct / total
        results.append({
            "Model": model_name,
            "Task": config,
            "Accuracy": accuracy
        })

# データフレームに変換
df = pd.DataFrame(results)
print(df)

# print(df)
# CSVに保存
df.to_csv("blimp_evaluation_results_test.csv", index=False)

# print("評価結果を 'blimp_evaluation_results.csv' に保存しました。")
