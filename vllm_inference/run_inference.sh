
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
CKPT=$1
DOMAIN=$2
MODEL_TYPE=$3
OUTPUT_DIR=$4
RESULTS_DIR=$5
OTHER_OPT=$6 # For debugging: you can append the '--stop 10' flag to watch all the intermediate input/output of the first 10 data examples

if [ ${DOMAIN} == 'med' ]; then
    TASK_array=(
        'SLAKE'
        "PathVQA"
        'VQA_RAD'
        "PMC_VQA"
    )
elif [ ${DOMAIN} == 'food' ]; then
    TASK_array=(
        "Recipe1M"
        "Nutrition5K"
        "FoodSeg103"
        "Food101"
    )
else
    TASK_array=(
        ${DOMAIN}
    )
fi

echo "Prepare Code for Domain: ${DOMAIN}, Model type: ${MODEL_TYPE}"

for j in "${!TASK_array[@]}"; do 
    TASK=${TASK_array[j]}
    echo "TASK: ${TASK}"
    echo "OUTPUT_DIR: ${OUTPUT_DIR}/${TASK}" # save outputs for every single task
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python inference.py \
            --model_weight_path ${CKPT}  \
            --task ${TASK} \
            --cuda_device ${GPULIST[$IDX]} \
            --output_dir ${OUTPUT_DIR}/${TASK} \
            --remove_cache \
            --data_parallel_size ${CHUNKS} \
            --model_type ${MODEL_TYPE} ${OTHER_OPT} &
    done

    wait

    echo 'inference done'

    python merge_predictions.py \
        --output_dir ${OUTPUT_DIR}/${TASK} \
        ${OTHER_OPT}

    python eval_predictions.py \
        --output_dir ${OUTPUT_DIR}/${TASK} \
        --model_path ${CKPT} \
        --task_name ${TASK} \
        --model_type ${MODEL_TYPE} \
        --eval_results_dir ${RESULTS_DIR} \
        ${OTHER_OPT}
done