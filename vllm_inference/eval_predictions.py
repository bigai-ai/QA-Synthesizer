import argparse
import json
from utils.task  import task_map
from tqdm import tqdm
import os
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='directory for saving the shards.')
parser.add_argument('--model_path', type=str)
parser.add_argument('--task_name', type=str)
parser.add_argument('--stop',type=int, default=None)
parser.add_argument('--model_type', choices = ['llava', 'qwen2_vl', 'mllama', 'llama'])
parser.add_argument('--eval_results_dir', type=str)

args = parser.parse_args()


if args.stop is not None:
    print(f'debug mode...')
    args.output_dir='/tmp/test_syn'

print(f'args.task_name: {args.task_name}')
task_cls = task_map.cls_dic[args.task_name](args.model_type)

if task_cls.enable_eval:
    metadata_list = []
    out_path = f'{args.output_dir}.jsonl'
    with open(out_path, 'r', encoding='utf8') as f:
        jsonls = f.read().strip().split('\n')
        for jsonl in tqdm(jsonls):
            metadata_list.append(json.loads(jsonl))

    print(f'eval: {out_path}')
    metadata_list, results = task_cls.evaluate(metadata_list, stop=args.stop)
    results_file = os.path.join(args.eval_results_dir, f'{args.task_name}.txt')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, "a") as f:
        info = {'pred_file': out_path,
                'model': str(args.model_path),
                'scores': results}
        f.write(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2) + "\n")
    print(f'write results to: {results_file}')

    # re-save metadatalist to save the score for each individual entry
    print(f're-saving scored metadatalist to {out_path}...')
    os.remove(out_path)
    with jsonlines.open(out_path,mode='a') as writer:
        for doc in metadata_list:
            writer.write(doc)
    print(f'saved jsonl to: {out_path}')

print('done')