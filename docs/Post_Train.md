# Post-Training General MLLMs
## Download or Reproduce Synthetic Training Datasets

You may follow [Synthesis.md](./Synthesis.md) to reproduce our training datasets. 
Or you can skip it and download the resulting synthetic data from:

- [biomed-visual-instructions](https://huggingface.co/datasets/AdaptLLM/biomed-visual-instructions)
- [food-visual-instructions](https://huggingface.co/datasets/AdaptLLM/food-visual-instructions)

## To post-train LLava-v1.6-Llama3-8B
Use the `adamllm` environment to post-train LLava-v1.6-Llama3-8B ([the open-source version](https://huggingface.co/Lin-Chen/open-llava-next-llama3-8b))

```bash
cd QA-Synthesizer
conda activate adamllm

# Biomedicine domain, using PMC^Refined caption
DOMAIN=biomed
DATASET=PATH_TO/biomed-visual-instructions/image_caption_and_synthetic_task.json
IMAGE_FOLDER=PATH_TO/biomed-visual-instructions/images 

bash ./scripts/post_train_mllm.sh ${DOMAIN} ${DATASET} ${IMAGE_FOLDER}

# Food domain
DOMAIN=food
DATASET=PATH_TO/food-visual-instructions/image_caption_and_synthetic_task.json
IMAGE_FOLDER=PATH_TO/food-visual-instructions/images 

bash ./scripts/post_train_mllm.sh ${DOMAIN} ${DATASET} ${IMAGE_FOLDER}

conda deactivate
```

## To post-train Qwen2-VL-2B-Instruct and Llama-3.2-11B-Vision-Instruct

### Update Dataset Information  

1. **Edit `dataset_info.json`**:  
Update the `file_name` field in the [dataset_info.json](../scripts/dataset_info.json) to point to the paths of your biomed and food training data.  

```json
{
    "biomed": {
        "file_name": "PATH_TO/biomed-visual-instructions/image_caption_and_synthetic_task.json",  // Replace with your file path
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    },
    "food": {
        "file_name": "PATH_TO/food-visual-instructions/image_caption_and_synthetic_task.json",  // Replace with your file path
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    }
}
```

2. **Copy `dataset_info.json` to the respective image folders**:  
Use the following commands to copy `dataset_info.json` to the image folders for both biomed and food domains:  

```bash
# Biomed domain
IMAGE_FOLDER=PATH_TO/biomed-visual-instructions/images 
cp ./scripts/dataset_info.json ${IMAGE_FOLDER}/ -v

# Food domain
IMAGE_FOLDER=PATH_TO/food-visual-instructions/images  
cp ./scripts/dataset_info.json ${IMAGE_FOLDER}/ -v
```  


### Set up environment for LLaMA-Factory and Qwen2-VL 
We utilize [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file) for experiments on Qwen2-VL-2B-Instruct and Llama-3.2-11B-Vision-Instruct.

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

conda create -n llama-factory python=3.10 -y
conda activate llama-factory
pip install -e ".[torch,metrics]"

# You may need to update the following packages for qwen-vl
# pip install trl==0.9.6 accelerate==0.34.0 git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830

pip install qwen-vl-utils
```

### To Post-Train Qwen2-VL-2B

```bash
BASE_MODEL=Qwen/Qwen2-VL-2B-Instruct
DATASET=food # Choose from [biomed, food]
IMAGE_FOLDER=PATH_TO/food-visual-instructions/images # Choose from [biomed-visual-instructions/images, food-visual-instructions/images]
BATCH_SIZE=8
GRADIENT_ACCU_STEPS=2
OUTPUT_PATH=./exp/${DATASET}-Qwen2-VL-2B

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -m torch.distributed.launch --use_env --nproc_per_node=8 --master_port=12345 src/train.py \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path ${BASE_MODEL} \
    --dataset ${DATASET} \
    --template qwen2_vl \
    --finetuning_type full \
    --output_dir ${OUTPUT_PATH} \
    --overwrite_cache \
    --warmup_ratio 0.1 \
    --weight_decay 0.1 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
    --ddp_timeout 180000000 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --cutoff_len 6144 \
    --save_steps 500 \
    --num_train_epochs 1 \
    --bf16 \
    --report_to none \
    --save_total_limit 1 \
    --preprocessing_num_workers 32 \
    --dataset_dir ${IMAGE_FOLDER}

# Copy configuration files for task evaluation
cp ${BASE_MODEL}/chat_template.json ${OUTPUT_PATH}/chat_template.json -v
cp ${BASE_MODEL}/merges.txt ${OUTPUT_PATH}/merges.txt -v
cp ${BASE_MODEL}/tokenizer_config.json ${OUTPUT_PATH}/tokenizer_config.json -v
cp ${BASE_MODEL}/tokenizer.json ${OUTPUT_PATH}/tokenizer.json -v
cp ${BASE_MODEL}/vocab.json ${OUTPUT_PATH}/vocab.json -v
```

### To Post-Train Llama-3.2-11B-Vision-Instruct

```bash
BASE_MODEL=meta-llama/Llama-3.2-11B-Vision-Instruct
DATASET=food # Choose from [biomed, food]
IMAGE_FOLDER=PATH_TO/food-visual-instructions/images # Choose from [biomed-visual-instructions/images, food-visual-instructions/images]
BATCH_SIZE=4
GRADIENT_ACCU_STEPS=4
OUTPUT_PATH=./exp/${DATASET}-Llama-3.2-11B-Vision

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -m torch.distributed.launch --use_env --nproc_per_node=8 --master_port=12345 src/train.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path ${BASE_MODEL} \
    --dataset ${DATASET} \
    --template llama3_vl \
    --finetuning_type full \
    --output_dir ${OUTPUT_PATH} \
    --overwrite_cache \
    --warmup_ratio 0.1 \
    --weight_decay 0.1 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
    --ddp_timeout 180000000 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --cutoff_len 6144 \
    --save_steps 500 \
    --num_train_epochs 1 \
    --bf16 \
    --report_to none \
    --save_total_limit 1 \
    --preprocessing_num_workers 32 \
    --dataset_dir ${IMAGE_FOLDER}

# Copy configuration files for task evaluation
cp ${BASE_MODEL}/chat_template.json ${OUTPUT_PATH}/chat_template.json -v
cp ${BASE_MODEL}/special_tokens_map.json ${OUTPUT_PATH}/special_tokens_map.json -v
cp ${BASE_MODEL}/tokenizer_config.json ${OUTPUT_PATH}/tokenizer_config.json -v
cp ${BASE_MODEL}/tokenizer.json ${OUTPUT_PATH}/tokenizer.json -v
```