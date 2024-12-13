from vllm import LLM, SamplingParams
import argparse
import os
from utils.cache_util import BufferedJsonWriter
from utils.task  import task_map
from tqdm import tqdm
from more_itertools import distribute
from transformers import AutoProcessor

parser = argparse.ArgumentParser()
parser.add_argument('--model_weight_path', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--stop',type=int, default=None)
parser.add_argument('--remove_cache', action='store_true', help='remove the cached files')
parser.add_argument('--data_parallel_size', type=int, default=1)
parser.add_argument('--cuda_device', type=int, default=0)
parser.add_argument('--task_name', type=str)
parser.add_argument('--model_type', choices = ['llava', 'qwen2_vl', 'mllama', 'llama'])

args = parser.parse_args()
task_cls = task_map.cls_dic[args.task_name](args.model_type)

#==== debug task args ===
print(f'### task_name: {args.task_name}')
print(f'task_cls.model_type: {task_cls.model_type}')
print(f'task_cls.stop_tokens: {task_cls.stop_tokens}')
print(f'task_cls.max_tokens: {task_cls.max_tokens}')
print(f'task_cls.skip_special_tokens: {task_cls.skip_special_tokens}')
print(f'task_cls.max_model_len: {task_cls.max_model_len}')

sampling_params = SamplingParams(temperature=0, max_tokens=task_cls.max_tokens, skip_special_tokens=task_cls.skip_special_tokens, stop=task_cls.stop_tokens)

ds = task_cls.get_dataset()

processor = None
process_vision_info = None
if args.model_type == 'llava':
    llm = LLM(model=args.model_weight_path, max_model_len=task_cls.max_model_len)
elif args.model_type == 'qwen2_vl':
    from qwen_vl_utils import process_vision_info
    llm = LLM(model=args.model_weight_path, max_model_len=task_cls.max_model_len)
    processor = AutoProcessor.from_pretrained(args.model_weight_path)
elif args.model_type == 'mllama':
    # add constraints to avoid oom for mllama-11b, ref: https://github.com/vllm-project/vllm/issues/9163#issuecomment-2400778274
    llm = LLM(model=args.model_weight_path, max_model_len=task_cls.max_model_len, dtype="bfloat16", gpu_memory_utilization=0.85, enforce_eager=True, max_num_seqs=20)
    processor = AutoProcessor.from_pretrained(args.model_weight_path)
elif args.model_type == 'llama':
    llm = LLM(model=args.model_weight_path, max_model_len=task_cls.max_model_len)

# assign identical index
for id, entry in enumerate(ds):
    entry['syn_id'] = id

if args.stop is not None:
    print(f'debug mode...')
    args.output_dir='/tmp/test_syn'
    ds = ds[:args.stop]

# We pass every 500 entries to llm.generate and write them to the cache
chunk_size = 500 if args.stop is None else 500

def run_non_ddp_inference_one_model(split, rank=0):
    print(f'cur rank: {rank}, infer on {len(split)} prompts')
    cached_file_path = f"{args.output_dir}tmp_process-{rank}.bin"
    os.makedirs(os.path.dirname(cached_file_path), exist_ok=True)
    print(f'cur rank: {rank}, cached_file_path: {cached_file_path}')

    with BufferedJsonWriter(cached_file_path, buffer_size=1) as buffer:
        cached_size = 0
        if os.path.exists(cached_file_path):
            if args.remove_cache:
                os.remove(cached_file_path)
                print(f"cur rank: {rank}, {cached_file_path} removed successfully")
            else:
                cached_size = buffer.get_cached_size()
                print(f'cur rank: {rank}, continue from {cached_file_path}')
        print(f'cur rank: {rank}, cached_size = {cached_size}...')

        assert cached_size % chunk_size == 0, f'cur rank: {rank}, we save the outputs every chunk_size, so the cached_size should be multiple of chunk_size'

        silent = False if rank == 0 else True # we only show progress on rank 0
        for start_index in tqdm(range(0, len(split), chunk_size), disable = silent):
            if start_index < cached_size: continue
            cur_split = split[start_index: start_index + chunk_size]
            cur_prompts = [task_cls.get_prompt(line, args.stop, silent=True, processor=processor, process_vision_info=process_vision_info) for line in cur_split]
            cur_prompts = [entry for entry in cur_prompts if entry is not None]
            try:
                outputs = llm.generate(cur_prompts, sampling_params, use_tqdm=False)
            except Exception as e:
                print(e)
                buffer.write([])
                continue
            metadata_list = []
            id = 0
            for metadata in cur_split:
                task_prompt = task_cls.get_prompt(metadata, silent=True, processor=processor, process_vision_info=process_vision_info)
                if task_prompt is None:
                    metadata.update({'pred': None})
                    if 'image' in metadata and not isinstance(metadata['image'], str):
                        # To avoid the `TypeError: Object of type PngImageFile is not JSON serializable` when saving the data
                        metadata.pop('image')
                    metadata_list.append(metadata)
                    continue
                output = outputs[id]
                if metadata['syn_id'] % 1000==0 or args.stop is not None:
                    # For debugging, print input and output details every 1000 examples or when 'stop' is triggered.
                    task_cls.get_prompt(metadata, args.stop, silent=False, processor=processor, process_vision_info=process_vision_info)
                    print(f'pred: {output.outputs[0].text}', flush=True)
                id += 1
                 
                if args.model_type in ['llava', 'qwen_vl', 'mllama'] and output.prompt[-10:] != task_prompt['prompt'][-10:]:
                    print(f'output.prompt: {output.prompt[-10:]} does not fit for task_prompt: {task_prompt["prompt"][-10:]}')
                    metadata.update({'pred': None})
                    metadata_list.append(metadata)
                    continue
                if args.model_type == 'llama' and output.prompt[-10:] != task_prompt[-10:]:
                    print(f'output.prompt: {output.prompt[-10:]} does not fit for task_prompt: {task_prompt[-10:]}')
                    metadata.update({'pred': None})
                    metadata_list.append(metadata)
                    continue
                metadata.update({'pred': output.outputs[0].text})

                if 'image' in metadata and not isinstance(metadata['image'], str):
                    # To avoid the `TypeError: Object of type PngImageFile is not JSON serializable` when saving the data
                    metadata.pop('image')
                metadata_list.append(metadata)
            # we set the buffere_size = 1, so `write_outputs_to_cache_path` would happen every time we call buffer.write(outputs)
            # see ./utils/cache_util.py
            assert id == len(cur_prompts) and id == len(outputs), f'id: {id} != len(cur_prompts): {len(cur_prompts)} != len(outputs): {len(outputs)}'
            buffer.write(metadata_list)
    print(f'cur rank: {rank}, saved all the outputs to {cached_file_path}')


if args.data_parallel_size > 1:
    print('ddp_mode...')
    # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
    # interleaved important to balance context lengths across workers
    sharded_splits = [list(x) for x in distribute(args.data_parallel_size, ds)]
    run_non_ddp_inference_one_model(sharded_splits[args.cuda_device], rank=args.cuda_device)
else:
    print('non_ddp_mode...')
    run_non_ddp_inference_one_model(ds)

print('generating done')