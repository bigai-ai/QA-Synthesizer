import numpy as np
from rouge import Rouge
import random

# copied from https://github.com/microsoft/LMOps/blob/main/uprise/src/utils/metric.py
def rouge(labels, preds, return_list=False):
    r1s, r2s, rls = [], [], []
    r = Rouge()
    for i in range(len(labels)):
        try:
            scores = r.get_scores(preds[i], labels[i])[0]
            r1s.append(scores["rouge-1"]["f"])
            r2s.append(scores["rouge-2"]["f"])
            rls.append(scores["rouge-l"]["f"])
        except Exception as e:
            r1s.append(0)
            r2s.append(0)
            rls.append(0)
    if return_list:
        return rls
    r1 = sum(r1s) / len(r1s)
    r2 = sum(r2s) / len(r2s)
    rl = sum(rls) / len(rls)
    return r1, r2, rl

# modified based on https://github.com/MMMU-Benchmark/MMMU/blob/main/eval/utils/eval_utils.py
# ----------- Process Multi-choice -------------
def parse_multi_choice_response(response, all_choices, index2ans, random_seed):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    random_flag = False # whether is random selected answer
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " " # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices: # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)
    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False # it's content ans.
    
    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.Random(random_seed).choice(all_choices)
        random_flag = True
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack: 
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index) # -1 will be ignored anyway
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else: # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index, random_flag

def parse_multi_label_response(response: str, index2ans: dict):
    """
    Parse the prediction from the generated response.
    """
    # to avoid partial match
    # for example, a sentence having egg tart would match both egg and egg tart, however, only egg tart should be matched.
    # so we sort the index2ans by length and match the longer first
    # if one phrase is matched, we replace the matched phrase with '' in the response
    index2ans = dict(sorted(index2ans.items(), key=lambda item: len(item[1]), reverse=True))
    candidates = []
    response = response.lower()
    for index, ans in index2ans.items():
        if ans.lower() in response:
            candidates.append(index)
            response = response.replace(ans.lower(), '')
    return candidates


from sklearn.metrics import recall_score, f1_score
def compute_multi_label_scores(label_indices: list, pred_indices: list, category_index_start: int, category_index_end: int, metric_name: str):

    # Convert to binary vectors
    y_true = [1 if i in label_indices else 0 for i in range(category_index_start, category_index_end + 1)]
    y_pred = [1 if i in pred_indices else 0 for i in range(category_index_start, category_index_end + 1)]

    # Calculate all metrics
    if metric_name == 'recall': 
        recall = recall_score(y_true, y_pred, zero_division=1)
        return recall * 100
    elif metric_name == 'f1':
        f1 = f1_score(y_true, y_pred, zero_division=1)
        return f1 * 100