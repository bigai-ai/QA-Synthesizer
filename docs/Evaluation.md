# Domain-Specific Task Evaluation
We provide all the resources necessary to reproduce our results and evaluate any MLLMs compatible with [vLLM](https://github.com/vllm-project/vllm).
## Model Zoo

| Model                                                                       | Repo ID in HF 🤗                           | Domain       | Base Model              | Training Data                                                                                  | Evaluation Benchmark |
|:----------------------------------------------------------------------------|:--------------------------------------------|:--------------|:-------------------------|:------------------------------------------------------------------------------------------------|-----------------------|
| [AdaMLLM-med-2B](https://huggingface.co/AdaptLLM/biomed-Qwen2-VL-2B-Instruct) | AdaptLLM/biomed-Qwen2-VL-2B-Instruct     | Biomedicine  | Qwen2-VL-2B-Instruct    | [biomed-visual-instructions](https://huggingface.co/datasets/AdaptLLM/biomed-visual-instructions) | [biomed-VQA-benchmark](https://huggingface.co/datasets/AdaptLLM/biomed-VQA-benchmark)                   |
| [AdaMLLM-food-2B](https://huggingface.co/AdaptLLM/food-Qwen2-VL-2B-Instruct) | AdaptLLM/food-Qwen2-VL-2B-Instruct     | Food  | Qwen2-VL-2B-Instruct    | [food-visual-instructions](https://huggingface.co/datasets/AdaptLLM/food-visual-instructions) | [food-VQA-benchmark](https://huggingface.co/datasets/AdaptLLM/food-VQA-benchmark)                   |
| [AdaMLLM-med-8B](https://huggingface.co/AdaptLLM/biomed-LLaVA-NeXT-Llama3-8B) | AdaptLLM/biomed-LLaVA-NeXT-Llama3-8B     | Biomedicine  | open-llava-next-llama3-8b    | [biomed-visual-instructions](https://huggingface.co/datasets/AdaptLLM/biomed-visual-instructions) | [biomed-VQA-benchmark](https://huggingface.co/datasets/AdaptLLM/biomed-VQA-benchmark)                   |
| [AdaMLLM-food-8B](https://huggingface.co/AdaptLLM/food-LLaVA-NeXT-Llama3-8B) |AdaptLLM/food-LLaVA-NeXT-Llama3-8B     | Food  | open-llava-next-llama3-8b    | [food-visual-instructions](https://huggingface.co/datasets/AdaptLLM/food-visual-instructions) |  [food-VQA-benchmark](https://huggingface.co/datasets/AdaptLLM/food-VQA-benchmark)                   |
| [AdaMLLM-med-11B](https://huggingface.co/AdaptLLM/biomed-Llama-3.2-11B-Vision-Instruct) | AdaptLLM/biomed-Llama-3.2-11B-Vision-Instruct     | Biomedicine  | Llama-3.2-11B-Vision-Instruct    | [biomed-visual-instructions](https://huggingface.co/datasets/AdaptLLM/biomed-visual-instructions) | [biomed-VQA-benchmark](https://huggingface.co/datasets/AdaptLLM/biomed-VQA-benchmark)                   |
| [AdaMLLM-food-11B](https://huggingface.co/AdaptLLM/food-Llama-3.2-11B-Vision-Instruct) | AdaptLLM/food-Llama-3.2-11B-Vision-Instruct     | Food | Llama-3.2-11B-Vision-Instruct    | [food-visual-instructions](https://huggingface.co/datasets/AdaptLLM/food-visual-instructions) |  [food-VQA-benchmark](https://huggingface.co/datasets/AdaptLLM/food-VQA-benchmark)                   |


## Task Datasets
To simplify the evaluation on domain-specific tasks, we have uploaded the templatized test sets for each task:

- [biomed-VQA-benchmark](https://huggingface.co/datasets/AdaptLLM/biomed-VQA-benchmark)
- [food-VQA-benchmark](https://huggingface.co/datasets/AdaptLLM/food-VQA-benchmark)

The dataset loading script is embedded in the inference code, so you can directly run the following commands to evaluate MLLMs.

## Evaluate Any MLLM Compatible with vLLM

Our code can directly evaluate models such as LLaVA-v1.6 ([open-source version](https://huggingface.co/Lin-Chen/open-llava-next-llama3-8b)), Qwen2-VL, and Llama-3.2-Vision. To evaluate other MLLMs, refer to [this guide](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_vision_language.py) for modifying the `BaseTask` class in the [vllm_inference/utils/task.py](../vllm_inference/utils/task.py) file. Feel free to reach out to us for assistance!

### Setup
Install vLLM with `pip` or [from source](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source).

As recommended in the official vLLM documentation, install vLLM in a **fresh new** conda environment:

```bash
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm  # Ensure vllm>=0.6.2 for compatibility with llama3.2; if llama-3.2 is not used, vllm==0.6.1 is sufficient
```

```bash
cd QA-Synthesizer/vllm_inference
RESULTS_DIR=./eval_results  # Directory for saving evaluation scores
```

### Biomedicine Domain

```bash
# Choose from ['med', 'PMC_VQA', 'VQA_RAD', 'SLAKE', 'PathVQA']
# 'med' runs inference on all biomedicine tasks; others run on a single task
DOMAIN='med'

# 1. LLaVA-v1.6-8B
MODEL_TYPE='llava'
MODEL=AdaptLLM/biomed-LLaVA-NeXT-Llama3-8B  # HuggingFace repo ID for AdaMLLM-med-8B
OUTPUT_DIR=./output/AdaMLLM-med-LLaVA-8B_${DOMAIN}

# Run inference with data parallelism; adjust CUDA devices as needed
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_inference.sh ${MODEL} ${DOMAIN} ${MODEL_TYPE} ${OUTPUT_DIR} ${RESULTS_DIR}

# 2. Qwen2-VL-2B
MODEL_TYPE='qwen2_vl'
MODEL=Qwen/Qwen2-VL-2B-Instruct  # HuggingFace repo ID for Qwen2-VL
OUTPUT_DIR=./output/Qwen2-VL-2B-Instruct_${DOMAIN}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_inference.sh ${MODEL} ${DOMAIN} ${MODEL_TYPE} ${OUTPUT_DIR} ${RESULTS_DIR}

MODEL=AdaptLLM/biomed-Qwen2-VL-2B-Instruct  # HuggingFace repo ID for AdaMLLM-med-2B
OUTPUT_DIR=./output/AdaMLLM-med-Qwen-2B_${DOMAIN}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_inference.sh ${MODEL} ${DOMAIN} ${MODEL_TYPE} ${OUTPUT_DIR} ${RESULTS_DIR}

# 3. Llama-3.2-11B
MODEL_TYPE='mllama'
MODEL=meta-llama/Llama-3.2-11B-Vision-Instruct  # HuggingFace repo ID for Llama3.2
OUTPUT_DIR=./output/Llama-3.2-11B-Vision-Instruct_${DOMAIN}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_inference.sh ${MODEL} ${DOMAIN} ${MODEL_TYPE} ${OUTPUT_DIR} ${RESULTS_DIR}

MODEL=AdaptLLM/biomed-Llama-3.2-11B-Vision-Instruct  # HuggingFace repo ID for AdaMLLM-11B
OUTPUT_DIR=./output/AdaMLLM-med-Llama3.2-11B_${DOMAIN}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_inference.sh ${MODEL} ${DOMAIN} ${MODEL_TYPE} ${OUTPUT_DIR} ${RESULTS_DIR}
```

### Food Domain

```bash
# Choose from ['food', 'Recipe1M', 'Nutrition5K', 'Food101', 'FoodSeg103']
# 'food' runs inference on all food tasks; others run on a single task
DOMAIN='food'

# 1. LLaVA-v1.6-8B
MODEL_TYPE='llava'
MODEL=AdaptLLM/food-LLaVA-NeXT-Llama3-8B  # HuggingFace repo ID for AdaMLLM-food-8B
OUTPUT_DIR=./output/AdaMLLM-food-LLaVA-8B_${DOMAIN}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_inference.sh ${MODEL} ${DOMAIN} ${MODEL_TYPE} ${OUTPUT_DIR} ${RESULTS_DIR}

# 2. Qwen2-VL-2B
MODEL_TYPE='qwen2_vl'
MODEL=Qwen/Qwen2-VL-2B-Instruct  # HuggingFace repo ID for Qwen2-VL
OUTPUT_DIR=./output/Qwen2-VL-2B-Instruct_${DOMAIN}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_inference.sh ${MODEL} ${DOMAIN} ${MODEL_TYPE} ${OUTPUT_DIR} ${RESULTS_DIR}

MODEL=AdaptLLM/food-Qwen2-VL-2B-Instruct  # HuggingFace repo ID for AdaMLLM-food-2B
OUTPUT_DIR=./output/AdaMLLM-food-Qwen-2B_${DOMAIN}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_inference.sh ${MODEL} ${DOMAIN} ${MODEL_TYPE} ${OUTPUT_DIR} ${RESULTS_DIR}

# 3. Llama-3.2-11B
MODEL_TYPE='mllama'
MODEL=meta-llama/Llama-3.2-11B-Vision-Instruct  # HuggingFace repo ID for Llama3.2
OUTPUT_DIR=./output/Llama-3.2-11B-Vision-Instruct_${DOMAIN}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_inference.sh ${MODEL} ${DOMAIN} ${MODEL_TYPE} ${OUTPUT_DIR} ${RESULTS_DIR}

MODEL=AdaptLLM/food-Llama-3.2-11B-Vision-Instruct  # HuggingFace repo ID for AdaMLLM-food-11B
OUTPUT_DIR=./output/AdaMLLM-food-Llama3.2-2B_${DOMAIN}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_inference.sh ${MODEL} ${DOMAIN} ${MODEL_TYPE} ${OUTPUT_DIR} ${RESULTS_DIR}
```

## Results

The evaluation results are stored in `./eval_results`, and the model prediction outputs are in `./output`.