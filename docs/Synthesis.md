# Visual Instruction Synthesis  

## 1. Fine-Tuning Visual Instruction Synthesizer  
We fine-tune a unified visual instruction synthesizer that generates diverse tasks based on image-caption pairs across various domains.  

The following steps reproduce our visual instruction synthesizer. Alternatively, you can skip these steps and download our synthesizer from [AdaptLLM/visual-instruction-synthesizer](https://huggingface.co/AdaptLLM/visual-instruction-synthesizer).  

Fine-tuning steps:  
```bash  
Coming soon...  
```  

## 2. Task Synthesis for Target Domain  
We use the synthesizer to generate task triplets from image-caption pairs in the target domain, followed by consistency-based data filtering to enhance data quality.  

The following steps reproduce our data. You can also skip them and download the resulting synthetic data (including `image_caption_and_synthetic_task.json` and `images`) from:
- [biomed-visual-instructions](https://huggingface.co/datasets/AdaptLLM/biomed-visual-instructions)  
- [food-visual-instructions](https://huggingface.co/datasets/AdaptLLM/food-visual-instructions)  

#### Setup  
```bash  
conda activate vllm  
cd QA-Synthesizer/vllm_inference  
SYNTHESIZER=AdaptLLM/visual-instruction-synthesizer  # Path to the synthesizer  
CONSISTENCY_CHECKER=meta-llama/Meta-Llama-3-8B  # Language model for consistency checks  
```  

#### **Quick Try with Data Samples** 
We have included a few [data samples](../data_samples) in this repository for a quick try:
```bash  
IMAGE_CAPTION='../data_samples/image_caption_pairs.json'  # Path to the image-caption pairs 
IMAGE_FOLDER='../data_samples/images'  # Path to the image folder  
OUTPUT_DIR='../data_samples/'  # Output directory for synthesized data  

# Run synthesis with data parallelism; adjust CUDA devices as needed:  
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_synthesis.sh ${SYNTHESIZER} ${CONSISTENCY_CHECKER} ${IMAGE_CAPTION} ${IMAGE_FOLDER} ${OUTPUT_DIR}  
```  

#### **Biomedicine** 
1. download the `image_caption_pairs.json` file and `images` from [AdaptLLM/biomed-visual-instructions](https://huggingface.co/datasets/AdaptLLM/biomed-visual-instructions)

2. Then run
  ```bash  
  IMAGE_CAPTION="./biomed-visual-instructions/image_caption_pairs.json"  
  IMAGE_FOLDER="./biomed-visual-instructions/images"  
  OUTPUT_DIR="./biomed-visual-instructions"  

  CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_synthesis.sh ${SYNTHESIZER} ${CONSISTENCY_CHECKER} ${IMAGE_CAPTION} ${IMAGE_FOLDER} ${OUTPUT_DIR}  
  ```  

#### **Food**  
1. download the `image_caption_pairs.json` file and `images` from [AdaptLLM/food-visual-instructions](https://huggingface.co/datasets/AdaptLLM/food-visual-instructions)

2. Then run
  ```bash  
  IMAGE_CAPTION="./food-visual-instructions/image_caption_pairs.json"  
  IMAGE_FOLDER="./food-visual-instructions/images"  
  OUTPUT_DIR="./food-visual-instructions"  

  CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' bash run_synthesis.sh ${SYNTHESIZER} ${CONSISTENCY_CHECKER} ${IMAGE_CAPTION} ${IMAGE_FOLDER} ${OUTPUT_DIR}  
  ```  

The synthesized output for single-stage post-training will be saved at: `${OUTPUT_DIR}/image_caption_and_synthetic_task.json`  