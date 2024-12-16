"""
utils for processing synthetic instructions
"""
import random
import os
from collections import defaultdict
import sys
sys.path.append("./")
import process.read_compre_pt as rc_pt_utils
import copy
import json

def save_json(dic, file_path):
    with open(file_path, "w") as writer:
        writer.write(json.dumps(dic, indent=4) + "\n")
    print('saved json to: ', file_path)

# ====================== Get the list of instruction-response pairs from the prediction ===================
def parse_pred(pred):
    """Get the list of instruction-response pairs from the prediction"""
    QA_str_list = pred.split('<|start_header_id|>user<|end_header_id|>\n\n')
    
    if not pred.endswith('<|eot_id|>'):
        # means the last answer is incomplete
        QA_str_list = QA_str_list[:-1]

    QA_list = []
    
    for QA_str in QA_str_list:
        try:
            assert len(QA_str.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')) == 2, f'invalid QA string: {QA_str}'
            Q_str, A_str = QA_str.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
            Q_str, A_str = Q_str.strip(), A_str[:-len('<|eot_id|>')].strip()
            assert len(Q_str) > 0, f'invalid question string in QA_str: {QA_str}'
            assert len(A_str) > 0, f'invalid answer string in QA_str: {QA_str}'
            QA_list.append({'Q': Q_str, 'A': A_str})
        except Exception as e:
            pass
    
    # save as the format in Finetune_Custom_Data.md
    conversations = []
    for qa_entry in QA_list:
        conversations.append({
            "from": "human",
            "value": qa_entry['Q']
        })
        conversations.append({
            "from": "gpt",
            "value": qa_entry['A']
        })    
    return conversations

# ====================== process_entry and save as the format in Finetune_Custom_Data.md ===================
def cook_classified_entry(entry):
    label = entry['pred'].strip().lower()
    # use indicators of mmmu benchmarking code, ref: mmmu/utils/eval_utils.py, remove `is `
    indicators_of_keys = ['could be ', 'so ', 'thus ', 'therefore ', 'final ', 'answer ', 'result ']
    basic_pattern = copy.deepcopy(random.Random(entry['syn_id']).choice(rc_pt_utils.FEWSHOT_PATTERNS['qa']))
    cot_pattern = copy.deepcopy(random.Random(entry['syn_id']).choice(rc_pt_utils.FEWSHOT_PATTERNS["qa_w_cot"]))
    collected_qa = entry['collected_QA']
    if label == 'yes':
        flag = None
        for indicator in indicators_of_keys:
            sents = collected_qa['informative_A'].strip().lower().split('. ')
            if len(sents) == 0: break
            last_sentence = sents[-1]
            if indicator in last_sentence:
                flag = 'informative_A_has_indicator'
                break
        if flag != 'informative_A_has_indicator':
            for indicator in indicators_of_keys: 
                if indicator in collected_qa['precise_A'].lower():
                    
                    flag = 'precise_A_has_indicator'
                    break
        
        if flag == 'informative_A_has_indicator':
            # use informative answer alone as answer
            q_template, a_template = basic_pattern.inputs, basic_pattern.targets
            key_dict = {'question': collected_qa['Q'], 'answer': collected_qa['informative_A']}
        elif flag == 'precise_A_has_indicator':
            # use the combination of informative and precise answer as answer
            q_template, a_template = basic_pattern.inputs, basic_pattern.targets
            combined_answer = f"{collected_qa['informative_A']} {collected_qa['precise_A']}"
            key_dict = {'question': collected_qa['Q'], 'answer': combined_answer}
            
        else:
            # use cot template, and treat informative answer as cot
            q_template, a_template = cot_pattern.inputs, cot_pattern.targets
            key_dict = {'question': collected_qa['Q'], 'cot': collected_qa['informative_A'], 'answer': collected_qa['precise_A']}
    else:
        # dump for no or open
        return []

    question = q_template.format(**key_dict)
    answer = a_template.format(**key_dict)

    syn_conversation = [{
                            "from": "human",
                            "value": question
                        },
                        {
                            "from": "gpt",
                            "value": answer
                        }]
    return syn_conversation

def process_entry(id, entry, image_token, train_on_syn_only=False):
    if entry['pred'] is None:
        return None

    entry['id'] = id
    syn_convers = cook_classified_entry(entry)
    
    existed_convers = []
    if not train_on_syn_only:
        existed_messages = entry.pop('messages')
        question = existed_messages[0]["content"]

        # remove <image> in existed_convers, we will add image token below
        question = question.replace("<image>\n", '').replace("<image>", '')

        existed_convers = [{
                            "from": "human",
                            "value": question
                        },
                        {
                            "from": "gpt",
                            "value": existed_messages[1]["content"]
                        }]

    # we randomly place existed conversations (of image caption) at the start or the end
    existed_place = random.Random(id).choice(['start', 'end'])
    conversations = existed_convers + syn_convers if existed_place == 'start' else syn_convers + existed_convers

    if len(conversations) == 0:
        return None

    # prepend image to the input
    conversations[0]["value"] = f'{image_token}\n{conversations[0]["value"]}'
    entry["conversations"] = conversations
    if id == 0:
        # print debug info at the beginning
        print(f"[DEBUG INFO] entry['id']: {entry['id']}")
        print(f'train_on_syn_only: {train_on_syn_only}')
        print(f"entry['pred']: {entry['pred']}")
        print(f"existed_convers: {existed_convers}")
        print(f"syn_convers: {syn_convers}")
        print(f"existed_place: {existed_place}")
        print(f"entry['conversations']: {entry['conversations']}")
    
    return entry

from tqdm import tqdm
def _load_json_or_jsonl(data_path, list_data_dict, rank=0):
    if rank == 0:
        print(f'loading from data path: {data_path}')
    if data_path.endswith('json'):
        list_data_dict += json.load(open(data_path, "r"))
    elif data_path.endswith('jsonl'):
        with open(data_path, 'r', encoding='utf8') as f:
            jsonls = f.read().strip().split('\n')
        for jsonl in tqdm(jsonls):
            list_data_dict.append(json.loads(jsonl))
    else:
        raise Exception(f'Only support files ended with .json or .jsonl, invalid path: {data_path}')
    return list_data_dict

# ================= Train Instruction Synthesizer =================
merged_dict = {}
task_dict = {}
duplicate_task_dict = {}
final_dict = {}
duplicate = 0
caption_hint = "Describe the image."
precise_hint = "Answer with a precise response."
informative_hint = "Answer with an informative response."

def format_caption(caption):
    conversation = [{"from": "human",
                    "value": f"<image>\n{caption_hint}"},
                    {"from": "gpt",
                    "value": caption}]
    return conversation

def load_caption(merged_dict, caption_path='./data/allava_vflan/ALLaVA-Caption-VFLAN-4V.json'):
    caption_data = json.load(open(caption_path))
    for entry in caption_data:
        assert len(entry['conversations']) == 2 and entry['conversations'][1]['from'] == 'gpt', f"invalid caption conversation: {entry['conversations']}"
        image = os.path.basename(entry['image'])
        if image in merged_dict:
            # duplicate_caption
            continue
        merged_dict[image] = entry
        caption = entry.pop('caption')
        merged_dict[image]['conversations'] = format_caption(caption)
    return

def load_precise_qa(merged_dict, task_name_only=False, precise_qa_path='./data/allava_vflan/vflan_metadata.json'):
    precise_qa_data = json.load(open(precise_qa_path))
    for entry in precise_qa_data:
        assert len(entry['conversations']) == 2 and entry['conversations'][1]['from'] == 'gpt', f"invalid precise conversation: {entry['conversations']}"
        image = os.path.basename(entry['image'])
        if image not in merged_dict:
            continue
        if not task_name_only:
            if 'precise_qa' in merged_dict[image]:
                merged_dict[image]['precise_qa'] += entry['conversations']
            else:
                merged_dict[image]['precise_qa'] = entry['conversations']
        merged_dict[image]['task_name'] = entry['task_name']
    return

def load_informative_qa(merged_dict, informative_qa_path='./data/allava_vflan/ALLaVA-Instruct-VFLAN-4V.json'):
    informative_qa_data = json.load(open(informative_qa_path))
    for entry in informative_qa_data:
        assert len(entry['conversations']) == 2 and entry['conversations'][1]['from'] == 'gpt', f"invalid informative conversation: {entry['conversations']}"
        image = os.path.basename(entry['image'])
        if image not in merged_dict:
            continue
        if 'informative_qa' in merged_dict[image]:
            merged_dict[image]['informative_qa'] += entry['conversations']
        else:
            merged_dict[image]['informative_qa'] = entry['conversations']
    return

def remove_image_token(question):
    if question.startswith("<image>\n"):
        question = question.replace("<image>\n", '')
    elif question.endswith("\n<image>"):
        question = question.replace("\n<image>", '')
    else:
        raise Exception(f'invalid image place in first QA question: {question}')
    return question

def format_qa(qa, mode):
    assert mode in ['precise', 'informative']
    conversations = []
    Hint = precise_hint if mode == 'precise' else informative_hint
    for idx in range(len(qa))[::2]:
        question = remove_image_token(qa[idx]['value'])
        answer = qa[idx+1]['value']
        conversations.append({'Q': f"{Hint}\n{question}", 'A': answer})
    return conversations

def replace_image_with_blank(id, entry, replace_with_blank_image_percent: int, blank_image_path: str = './assets/Blank.jpg'):
    """replace 5% of the images to blank image, so to force the synthesizer to learn the maping from text only"""
    assert replace_with_blank_image_percent >= 0 and replace_with_blank_image_percent <= 100
    replace = random.Random(id).choices([True, False], weights=(replace_with_blank_image_percent, 100 - replace_with_blank_image_percent), k=1)[0]
    if replace:
        entry['image'] = blank_image_path
    return entry

def load_syn_data(syn_mode, split='train', replace_with_blank_image_percent=10):
    assert syn_mode in ['precise', 'informative', 'precise+informative']

    merged_dict = {}
    
    load_caption(merged_dict)
    load_precise_qa(merged_dict)
    load_informative_qa(merged_dict)
    # collect task_name
    # load_precise_qa(merged_dict, task_name_only=True)
    idx = 0
    task_dict = defaultdict(list)
    for entry in merged_dict.values():
        if 'precise_qa' not in entry or 'informative_qa' not in entry:
            continue
        precise_conversations = format_qa(qa=entry.pop('precise_qa'), mode='precise')
        informative_conversations = format_qa(qa=entry.pop('informative_qa'), mode='informative')
        task_dict[entry['task_name']].append(entry)
        qa_pairs = precise_conversations + informative_conversations
        random.Random(idx).shuffle(qa_pairs)
        for qa in qa_pairs:
            entry['conversations'] += [{"from": "human", "value": qa['Q']},{"from": "gpt","value": qa['A']}]
        idx += 1
    task_num_dict = {k: len(v) for k, v in sorted(task_dict.items(), key=lambda item: len(item[1]), reverse=True)}

    # split to 90%:10% for train and validation
    train_list_dict = []
    dev_list_dict = []
    for task, entry_list in task_dict.items():
        random.Random(42).shuffle(entry_list)
        if replace_with_blank_image_percent > 0:
            entry_list = [replace_image_with_blank(id, entry, replace_with_blank_image_percent) for id, entry in enumerate(entry_list)]     

        train_list_dict += entry_list[:int(0.9 * len(entry_list))]
        dev_list_dict += entry_list[int(0.9 * len(entry_list)):]
        task_num_dict[task] = len(entry_list)

    random.Random(12345).shuffle(train_list_dict)
    random.Random(12345).shuffle(dev_list_dict)
    
    return train_list_dict if split == 'train' else dev_list_dict
