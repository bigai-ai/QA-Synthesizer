gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
SYNTHESIZER=$1  # AdaptLLM/visual-instruction-synthesizer
CONSISTENCY_CHECKER=$2  # meta-llama/Meta-Llama-3-8B
IMAGE_CAPTION=$3 # Path to the json file of image_caption pairs (in the ShareGPT format)
IMAGE_FOLDER=$4 # Path to the image folder
OUTPUT_DIR=$5
OTHER_OPT=$6    # you can add the "--stop 10" flag to watch all the intermediate input/output of the first 10 data examples

# 1. Synthesize `instruction-informative response-precise response` triplets 
echo "Synthesizing task triplets..."
echo "Output path: ${OUTPUT_DIR}/syn_task_triplets.jsonl"
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python inference.py \
        --model_weight_path ${SYNTHESIZER}  \
        --task syn_task_triplet \
        --data_path ${IMAGE_CAPTION} \
        --image_folder ${IMAGE_FOLDER} \
        --cuda_device ${GPULIST[$IDX]} \
        --output_dir ${OUTPUT_DIR}/syn_task_triplets \
        --remove_cache \
        --data_parallel_size ${CHUNKS} \
        --model_type 'llava' ${OTHER_OPT} &
done

wait

python merge_predictions.py \
    --output_dir ${OUTPUT_DIR}/syn_task_triplets \
    ${OTHER_OPT}

echo 'Synthesis done'

# 2. Consistency-based filter
echo "Conducting consistency-based filtering..."
echo "Output path: ${OUTPUT_DIR}/filtered_task_pairs.jsonl"
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python inference.py \
        --model_weight_path ${CONSISTENCY_CHECKER}  \
        --task consistency_filter \
        --data_path ${OUTPUT_DIR}/syn_task_triplets.jsonl \
        --image_folder ${IMAGE_FOLDER} \
        --cuda_device ${GPULIST[$IDX]} \
        --output_dir ${OUTPUT_DIR}/filtered_task_pairs \
        --remove_cache \
        --data_parallel_size ${CHUNKS} \
        --model_type 'llama' ${OTHER_OPT} &
done

wait

python merge_predictions.py \
    --output_dir ${OUTPUT_DIR}/filtered_task_pairs \
    ${OTHER_OPT}

echo 'Filter done'

python format_post_train_data.py \
	--filtered_task_pairs ${OUTPUT_DIR}/filtered_task_pairs.jsonl \
    --image_folder ${IMAGE_FOLDER} \
	--output_path ${OUTPUT_DIR}/image_caption_and_synthetic_task.json \
	${OTHER_OPT}

echo "Single-stage training data saved to: ${OUTPUT_DIR}/image_caption_and_synthetic_task.json"