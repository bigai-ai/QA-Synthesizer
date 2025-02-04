
DOMAIN=$1
DATASET=$2
IMAGE_FOLDER=$3
BATCH_SIZE=4
GRADIENT_ACCU_STEPS=4
SAVE_PATH=./exp/${DOMAIN}-LLaVA-v1.6-8B

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -m torch.distributed.launch --use_env --nproc_per_node=8 --master_port=12345 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path Lin-Chen/open-llava-next-llama3-8b \
    --version llava_llama_3 \
    --image_folder ${IMAGE_FOLDER}  \
    --vision_tower "/openai/clip-vit-large-patch14-336" \
    --mm_projector_type mlp2x_gelu \
    --unfreeze_mm_vision_tower True \
    --mm_vision_tower_lr 2e-6 \
    --image_aspect_ratio anyres \
    --group_by_modality_length True \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_patch_merge_type spatial_unpad \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 6144 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to none \
    --run_name ${SAVE_PATH} \
    --data_path ${DATASET}