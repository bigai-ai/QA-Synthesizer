import sys
import os
import utils.metric as metric
from PIL import Image, PngImagePlugin
import json
from tqdm import tqdm
from copy import deepcopy
from datasets import load_dataset
sys.path.append("./")
import collections
from utils.nutrition5k_ingredients import all_ingredients
from utils.llava_med.evaluate_metrics import calculate_f1score
from utils.llava_med.glossary import normalize_word
from utils.conversation import conv_templates

sys.path.append("../")
import process.syn_utils as syn_utils

class App:
    def __init__(self):
        self.cls_dic = {}

    def add(self, key):
        def adder(cls):
            self.cls_dic[key] = cls
            return cls

        return adder

task_map = App()
# usage: import task_map; cls=task_map.cls_dic[task_name]()

class BaseTask(object):
    def __init__(self, model_type):
        # default settings for llava-v1.6
        self.conv_mode = "llava_llama_3"
        self.image_token = '<|reserved_special_token_4|>'
        self.model_type = model_type
        
        if self.model_type in ['llava', 'mllama']:
            self.stop_tokens = ["<|end_of_text|>", "<|eot_id|>"] 
        elif self.model_type == 'qwen2_vl':
            self.stop_tokens = [] # according to qwen2_vl's official readme
        
        self.max_tokens = 1024 # max new tokens to generate
        self.skip_special_tokens = True # whether skip when decoding
        self.max_model_len = 6144 # = model training length
        
        self.enable_eval = True

        # copied from qwen's readme
        self.default_qwen2_vl_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "xxx", # replace with yours
                            "min_pixels": 224 * 224,
                            "max_pixels": 1280 * 28 * 28,
                        },
                        {"type": "text", "text": "xxx"}, # replace with yours
                    ],
                },
            ]
    
        # copied from llama_vl's readme
        self.default_mllama_messages = [{
                    "role": "user", 
                    "content": [
                            {"type": "image"},
                            {"type": "text", "text": "xxx"} # replace with yours
                    ]}]

    def get_image(self, entry):
        image_file = entry["image"]
        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert('RGB') # convert to avoid non-RGB images
        return image
    
    def get_prompt(self, entry, stop=None, silent=False, processor=None, process_vision_info=None, **kwargs):
        try:
            image = self.get_image(entry)
        except Exception as e:
            print(e)
            return None
        
        if self.model_type == 'llava':
            conv = conv_templates[self.conv_mode].copy()
            final_question = f"{self.image_token}\n{self.get_raw_question(entry)}"
            conv.append_message(conv.roles[0], final_question )
            conv.append_message(conv.roles[1], None)
            # NOTE: open-llava-next tokenizer would automatically add a bos token, which means there would be two bos at the start, 
            # this is strange but consistent with openllave's training setting, so we keep this setting when inference
            prompt = conv.get_prompt()
        
        elif self.model_type == 'qwen2_vl':
            messages = deepcopy(self.default_qwen2_vl_messages)
            messages[1]["content"][0]["image"] = image
            messages[1]["content"][1]["text"] = self.get_raw_question(entry)

            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image, _ = process_vision_info(messages) # _ is for video_inputs which we do not use
        
        elif self.model_type == 'mllama':
            messages = deepcopy(self.default_mllama_messages)
            messages[0]["content"][1]["text"] = self.get_raw_question(entry)
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        if (entry['syn_id'] % 1000 == 0 or stop is not None) and not silent:
            # For debugging, print input and output details every 1000 examples or when 'stop' is triggered.
            print(f"[DEBUG INFO] ID: {entry['syn_id']}")
            print(f"Entry: {entry}")
            
            if 'image' in entry and isinstance(entry['image'], str):
                print(f"Image Path: {os.path.join(self.local_image_folder, entry['image'])}")
            
            if self.model_type in ['qwen2_vl', 'mllama']:
                print(f"Messages: {messages}")
            
            print(f"Prompt: {prompt}")
            
        return {"prompt": prompt, "multi_modal_data": {"image": image}}


# ================================ Visual Instruction Synthesizer =====================================
@task_map.add("syn_task_triplet")
class syn_task_triplet(BaseTask):
    """synthesize `instruction-informative response-precise response` triplets from image-caption pairs"""
    def __init__(self, model_type):
        super().__init__(model_type)
        assert model_type == 'llava', "we only support llava-based synthesizer now"
        self.stop_tokens = ["<|end_of_text|>"]
        self.max_tokens = 512 # max new tokens to generate
        self.skip_special_tokens = False # do not skip because we need the <eot_id> for extracting qa pairs
        self.max_model_len = 4096 # 6144 -> 4096, constrain max model len for speed-up
        self.enable_eval = False
    
    def get_dataset(self, data_path, image_folder, **kwargs):
        """
        data_path is the path to the image_caption_pairs.json,
        where each entry is in the `ShareGPT` format:
        "images": [
            "image_xxx.jpg"
        ],
        "messages": [
            {
                "content": "<image>instruction",
                "role": "user"
            },
            {
                "content": "response",
                "role": "assistant"
            }
        ]
        """
        self.image_folder = image_folder 
        ds = json.load(open(data_path))
        return ds
        
    def get_prompt(self, entry, stop=None, silent=False, **kwargs):
        image_file = entry["images"][0]
        try:
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        except Exception as e:
            print(e)
            return None

        conv = conv_templates[self.conv_mode].copy()

        caption_question = f"{self.image_token}\n{syn_utils.caption_hint}"
        caption_answer = entry["messages"][1]["content"]
        conv.append_message(conv.roles[0], caption_question)
        conv.append_message(conv.roles[1], caption_answer)
        conv.append_message(conv.roles[0], None)

        prompt = conv.get_prompt()

        if (entry['syn_id'] % 1000 == 0 or stop is not None) and not silent:
            print(f"[DEBUG INFO] id: {entry['syn_id']}")
            print(f'entry: {entry}')
            print(f'image path: {os.path.join(self.image_folder, image_file)}')
            print(f'prompt: {prompt}')
            
        return {"prompt": prompt, "multi_modal_data": {"image": image}}
    
    def debug_info(self, metadata_list):
        for entry in metadata_list:
            print(f"## image path: {os.path.join(self.image_folder, entry['image'])}")
            print("caption: " + json.dumps(entry["conversations"][:2], indent=2) + "\n")
            print("groundtruth: " + json.dumps(entry["conversations"][2:], indent=2) + "\n")
            print("pred: " + json.dumps(syn_utils.parse_pred(entry["pred"])[0], indent=2) + "\n")
        return

@task_map.add("consistency_filter")
class consistency_filter(BaseTask):
    """Filter synthetic tasks based on response consistency"""
    def __init__(self, model_type):
        super().__init__(model_type)
        assert self.model_type == 'llama', "we use text-only language model for data fitering"

        prompt_path = './utils/consistency_filter_prompt.txt'
        self.prompt_template = open(prompt_path).read()

        self.max_tokens = 2 # max new tokens to generate
        self.skip_special_tokens = True
        self.stop_tokens = ["<|end_of_text|>"]
        self.max_model_len = 8192
        self.enable_eval = False
    
    def get_dataset(self, data_path, image_folder, stop, **kwargs):
        self.image_folder = image_folder # Image is not used for inference, but for debugging
        if stop is not None:
            data_path = '/tmp/test_syn.jsonl'
        ds = []
        with open(data_path, 'r', encoding='utf8') as f:
            jsonls = f.read().strip().split('\n')
            for jsonl in tqdm(jsonls):
                ds.append(json.loads(jsonl))
        return ds

    def get_prompt(self, entry, stop=None, silent=False, **kwargs):
        if entry['pred'] is None:
            return None
        pred_QAs = syn_utils.parse_pred(entry['pred'])

        precise_QAs = {}
        informative_QAs = {}
        precise_hint = f'{syn_utils.precise_hint}\n'
        informative_hint = f'{syn_utils.informative_hint}\n'

        collected_QA = None
        for idx in range(len(pred_QAs))[::2]:
            question = pred_QAs[idx]['value']
            answer = pred_QAs[idx+1]['value']
            if question.startswith(precise_hint):
                precise_q = question[len(precise_hint):]
                if precise_q in informative_QAs:
                    collected_QA = {
                    "Q": precise_q,
                    "precise_A": answer,
                    "informative_A": informative_QAs[precise_q]
                    }
                    break
                else:
                    precise_QAs[precise_q] = answer
            elif question.startswith(informative_hint):
                informative_q = question[len(informative_hint):]
                if informative_q in precise_QAs:
                    collected_QA = {
                    "Q": informative_q,
                    "precise_A": precise_QAs[informative_q],
                    "informative_A": answer
                    }
                    break
                else:
                    informative_QAs[informative_q] = answer
        
        if collected_QA is None:
            return None

        prompt = self.prompt_template.format(**collected_QA)
        entry['collected_QA'] = collected_QA

        # DEBUG INFO
        if (entry['syn_id'] % 1000 == 0 or stop is not None) and not silent:
            print(f"### id: {entry['syn_id']}")
            print("image: ", os.path.join(self.image_folder, entry['images'][0]))
            cur_prompt = prompt.split('\n\n## Question:')[-1]
            print(f"testing_prompt: {cur_prompt}")
        return prompt

# ======================================= Task Evaluation ======================================
class BaseHFTask(BaseTask):
    """evaluate tasks from huggingface repo of AdaMLLM: https://huggingface.co/AdaptLLM/Adapt-MLLM-to-Domains"""
    def __init__(self, model_type):
        super().__init__(model_type)
    
    def get_dataset(self, **kwargs):
        ds = list(load_dataset(f'AdaptLLM/{self.domain}-VQA-benchmark', self.task_name, split='test'))
        return ds

    def get_image(self, entry):
        image = entry["image"].convert('RGB')
        return image

    def get_raw_question(self, entry, **kwargs):
        return entry['input']

    def evaluate(self, metadata_list, stop=None):
        metadata_list = [self.evaluate_entry(metadata) for metadata in tqdm(metadata_list)]
        score = sum([entry[self.metric_name] for entry in metadata_list])/len(metadata_list)
        task_score_dict = {'test_num': len(metadata_list), self.metric_name: score}

        if stop is not None:
            self.debug_info(metadata_list[:5]) # show first 5 entries for debug
        return metadata_list, task_score_dict

    def debug_info(self, metadata_list):
        for entry in metadata_list:
            print(f"input: {entry['input']}")
            print(f"label: {entry['label']}")
            print(f"pred: {entry['pred']}")
            print(f"{self.metric_name}: {entry[self.metric_name]}")
        return

# ============= Food Tasks =============
@task_map.add("Recipe1M")
class Recipe1M(BaseHFTask):
    def __init__(self, model_type):
        super().__init__(model_type)
        self.domain = 'food'
        self.task_name = 'Recipe1M'
        self.metric_name = 'rl'

    def evaluate_entry(self, metadata):
        label, pred = metadata['label'], metadata['pred']
        r1, r2, rl = metric.rouge(labels=[label], preds=[pred])
        metadata['rl'] = rl*100
        return metadata

@task_map.add("Food101")
class Food101(BaseHFTask):
    def __init__(self, model_type):
        super().__init__(model_type)
        self.domain = 'food'
        self.task_name = 'Food101'
        self.metric_name = 'acc'

        self.max_tokens = 256 # 1024 -> 256 to avoid too much repetitive output
        map_path = './utils/food101_name_to_label_map.json'
        self.name_to_label_map = json.load(open(map_path))
        self.name_to_label_map = {key.replace('_', ' '): value for key, value in self.name_to_label_map.items()}
        self.label_to_name_map = {value: key for key, value in self.name_to_label_map.items()}
        self.all_choices = [str(index) for index in range(len(self.name_to_label_map))]

    def evaluate_entry(self, metadata):
        label_option, pred = self.name_to_label_map[metadata['label']], metadata['pred']
        pred_option, random_flag = metric.parse_multi_choice_response(pred, self.all_choices, self.label_to_name_map, metadata['syn_id'])
        acc = 1 if str(pred_option) == str(label_option) else 0
        metadata['label_option'] = label_option
        metadata['pred_option'] = pred_option
        metadata['random_selected'] = random_flag
        metadata['acc'] = acc * 100
        return metadata

@task_map.add("FoodSeg103")
class FoodSeg103(BaseHFTask):
    def __init__(self, model_type):
        super().__init__(model_type)
        self.domain = 'food'
        self.task_name = "FoodSeg103"

        self.max_tokens = 256 # 1024 -> 256 to avoid too much repetitive output
        map_path = './utils/foodSeg103_id2label.json'
        self.id2name_map = json.load(open(map_path))
        # 0 and 103 represent background and other ingredients
        self.id2name_map.pop("0")
        self.id2name_map.pop("103")
        self.id2name_map = {int(key): value for key, value in self.id2name_map.items()}
        self.name2id_map = {value: key for key, value in self.id2name_map.items()}
        self.metric_name = 'f1'

    def evaluate_entry(self, metadata):
        label_classes_on_image = [self.name2id_map[l] for l in metadata['label']]
        pred_classes_on_image = metric.parse_multi_label_response(metadata['pred'], index2ans = self.id2name_map)

        metadata[self.metric_name] = metric.compute_multi_label_scores(label_classes_on_image, pred_classes_on_image, 1, 102, metric_name=self.metric_name)
        metadata['pred_classes_on_image'] = sorted(pred_classes_on_image) # for readability
        metadata['pred_classes_names_on_image'] = [self.id2name_map[id] for id in metadata['pred_classes_on_image']]
        return metadata
    
@task_map.add("Nutrition5K")
class Nutrition5K(BaseHFTask):
    def __init__(self, model_type):
        super().__init__(model_type)
        self.domain = 'food'
        self.task_name = "Nutrition5K"
        self.metric_name = 'recall'

        self.max_tokens = 256 # 1024 -> 256 to avoid too much repetitive output
        self.id2name_map = dict(zip(range(0, len(all_ingredients)), all_ingredients))
        self.name2id_map = {value: key for key, value in self.id2name_map.items()}        

    def evaluate_entry(self, metadata):
        label_classes_on_image = [self.name2id_map[ing] for ing in metadata['label']]
        pred_classes_on_image = metric.parse_multi_label_response(metadata['pred'], index2ans = self.id2name_map)

        metadata[self.metric_name] = metric.compute_multi_label_scores(label_classes_on_image, pred_classes_on_image, 0, len(all_ingredients)-1, metric_name = self.metric_name)
        metadata['label_classes_on_image'] = sorted(label_classes_on_image) # sort for readability
        metadata['pred_classes_on_image'] = sorted(pred_classes_on_image) # sort for readability
        metadata['pred_classes_names_on_image'] = [self.id2name_map[id] for id in metadata['pred_classes_on_image']]
        return metadata

# ======================== BioMed ===========================
@task_map.add("PMC_VQA")
class PMC_VQA(BaseHFTask):
    def __init__(self, model_type):
        super().__init__(model_type)
        self.domain = 'biomed'
        self.task_name = "PMC-VQA"
        self.metric_name = 'acc'

        self.max_tokens = 1024  
        self.all_choices = ['A', 'B', 'C', 'D']
    
    def evaluate_entry(self, metadata):
        label_option, pred = metadata["label"], metadata['pred']
        label_to_name_map = {'A': metadata["A"], 
                            'B': metadata["B"], 
                            'C': metadata["C"], 
                            'D': metadata["D"]}
        
        pred_option, random_flag = metric.parse_multi_choice_response(pred, self.all_choices, label_to_name_map, metadata['syn_id'])
        acc = 1 if str(pred_option) == str(label_option) else 0
        metadata['label_option'] = label_option
        metadata['pred_option'] = pred_option
        metadata['random_selected'] = random_flag
        metadata['acc'] = acc * 100
        return metadata

@task_map.add("VQA_RAD")
class VQA_RAD(BaseHFTask):
    def __init__(self, model_type):
        super().__init__(model_type)
        self.domain = 'biomed'
        self.task_name = 'VQA_RAD'

    # modified based on https://github.com/microsoft/LLaVA-Med/blob/v1.0.0/llava/eval/run_eval.py
    def evaluate(self, metadata_list, stop=None):
        closed_scores = collections.defaultdict(list)
        f1_scores = collections.defaultdict(list)

        for entry in metadata_list:
            gt_value = entry['label'].lower()
            pred_value = entry['pred'].lower()

            gt_value = normalize_word(gt_value)
            pred_value = normalize_word(pred_value)

            if entry['answer_type'] == 'OPEN':
                f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
                f1_scores['recall'].append(recall)
                f1_scores['q_id'].append(entry['syn_id'])

                # we save metric score inside each entry for debug
                entry['score'] = {"recall": recall*100}

            elif entry['answer_type'] == 'CLOSED':
                # for close-ended question (Yes/No)
                closed_scores['q_id'].append(entry['syn_id'])
                if 'yes' in pred_value or 'no' in pred_value:
                    if gt_value in pred_value:
                        closed_scores['hit'].append(1)
                        entry['score'] = {'yes/no accuracy': 100}
                    else:
                        closed_scores['hit'].append(0)
                        entry['score'] = {'yes/no accuracy': 0}
                else:
                    closed_scores['hit'].append(0)
                    entry['score'] = {'yes/no accuracy': 0}                
        
        recall = sum(f1_scores['recall']) / len(f1_scores['recall'])
        closed_score = sum(closed_scores['hit']) / len(closed_scores['hit']) if len(closed_scores['hit']) != 0 else 0.0

        task_score_dict = { 'test_num': len(metadata_list),
                            'OPEN recall': recall*100,
                            'CLOSED yes/no accuracy': closed_score*100}

        if stop is not None:
            self.debug_info(metadata_list[:5])
    
        return metadata_list, task_score_dict

    def debug_info(self, metadata_list):
        for entry in metadata_list:
            print(f"input: {entry['input']}")
            print(f"label: {entry['label']}")
            print(f"pred: {entry['pred']}")
            print(f"score: {entry['score']}")
        return
        
@task_map.add("SLAKE")
class SLAKE(VQA_RAD):
    def __init__(self, model_type):
        super().__init__(model_type)
        self.task_name = 'SLAKE'

@task_map.add("PathVQA")
class PathVQA(VQA_RAD):
    def __init__(self, model_type):
        super().__init__(model_type)
        self.task_name = 'PathVQA'