"""
convert the data after the consistency-filter into our single-stage post-train format
"""
import sys
sys.path.append("../")
import process.syn_utils as syn_utils
import argparse
import os
from tqdm import tqdm
from PIL import Image
import json

parser = argparse.ArgumentParser()
parser.add_argument('--filtered_task_pairs', type=str, default='')
parser.add_argument('--image_folder', type=str, default='')
parser.add_argument('--output_path', type=str)
parser.add_argument('--train_on_syn_only', action='store_true', help='whether to remove the image-captioning task.')
parser.add_argument('--stop',type=int, default=None)

args = parser.parse_args()

if args.stop is not None:
    print(f'debug mode...')
    args.filtered_task_pairs='/tmp/test_syn.jsonl'
    args.output_path='/tmp/test_train_data.json'

ds = []
with open(args.filtered_task_pairs, 'r', encoding='utf8') as f:
    jsonls = f.read().strip().split('\n')
    for jsonl in tqdm(jsonls):
        ds.append(json.loads(jsonl))

id = 0
clean_ds = []
for entry in tqdm(ds):
    entry = syn_utils.process_entry(id=id, entry=entry, image_token='<image>', train_on_syn_only=args.train_on_syn_only)
    if entry is None:
      continue

    image_file = entry["images"][0]
    try:
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
    except Exception as e:
        print(e)
        continue
    clean_ds.append(entry)
    id += 1

# reformat to llama-factory's style
"""
factory example:
{
    "messages": [
      {
        "content": "<image>Who are they?",
        "role": "user"
      },
      {
        "content": "They're Kane and Gretzka from Bayern Munich.",
        "role": "assistant"
      }
    ],
    "images": [
      "mllm_demo_data/1.jpg"
    ]
}
"""
def reformat(entry):
    new_entry = {}
    # we place the image input at the beginning of instruction
    assert entry['conversations'][0]['value'].startswith('<image>\n')
    first_message = {
        "content": entry['conversations'][0]['value'],
        "role": "user"
      }
    new_entry['messages'] = [first_message]
    for ex in entry['conversations'][1:]:
      assert '<image>' not in ex['value']
      new_m = {
        'content': ex['value'],
        'role': "user" if ex['from'] == "human" else "assistant"
        }
      new_entry['messages'].append(new_m)
    return new_entry

factory_data = [reformat(entry) for entry in tqdm(clean_ds)]

syn_utils.save_json(factory_data, args.output_path)