# Adapting Multimodal Large Language Models to Domains via Post-Training

This repository provides models, code and data of our paper: [On Domain-Specific Post-Training for Multimodal Large Language Models](https://arxiv.org/abs/2411.19930).

We investigate domain adaptation of MLLMs through post-training, focusing on data synthesis, training pipelines, and task evaluation. 
**(1) Data Synthesis**: Using open-source models, we develop a visual instruction synthesizer that effectively generates diverse visual instruction tasks from domain-specific image-caption pairs. **Our synthetic tasks surpass those generated by manual rules, GPT-4, and GPT-4V in enhancing the domain-specific performance of MLLMs.** 
**(2) Training Pipeline**: While the two-stage training--initially on image-caption pairs followed by visual instruction tasks--is commonly adopted for developing general MLLMs, we apply a single-stage training pipeline to enhance task diversity for domain-specific post-training. 
**(3) Task Evaluation**: We conduct experiments in two domains, biomedicine and food, by post-training MLLMs of different sources and scales (e.g., Qwen2-VL-2B, LLaVA-v1.6-8B, Llama-3.2-11B), and then evaluating MLLM performance on various domain-specific tasks.


## Resouces
#### Domain-Specific Training Data
- [biomed-visual-instructions](https://huggingface.co/datasets/AdaptLLM/biomed-visual-instructions)
- [food-visual-instructions](https://huggingface.co/datasets/AdaptLLM/food-visual-instructions)


#### Domain-Specific Evaluation Benchmark

- [biomed-VQA-benchmark](https://huggingface.co/datasets/AdaptLLM/biomed-VQA-benchmark)
- [food-VQA-benchmark](https://huggingface.co/datasets/AdaptLLM/food-VQA-benchmark)

#### Domain-Specific Models
| Model                                                                       | Repo ID in HF 🤗                           | Domain       | Base Model              | Training Data                                                                                  | Evaluation Benchmark |
|:----------------------------------------------------------------------------|:--------------------------------------------|:--------------|:-------------------------|:------------------------------------------------------------------------------------------------|-----------------------|
| [AdaMLLM-med-2B](https://huggingface.co/AdaptLLM/biomed-Qwen2-VL-2B-Instruct) | AdaptLLM/biomed-Qwen2-VL-2B-Instruct     | Biomedicine  | Qwen2-VL-2B-Instruct    | [biomed-visual-instructions](https://huggingface.co/datasets/AdaptLLM/biomed-visual-instructions) | [biomed-VQA-benchmark](https://huggingface.co/datasets/AdaptLLM/biomed-VQA-benchmark)                   |
| [AdaMLLM-food-2B](https://huggingface.co/AdaptLLM/food-Qwen2-VL-2B-Instruct) | AdaptLLM/food-Qwen2-VL-2B-Instruct     | Food  | Qwen2-VL-2B-Instruct    | [food-visual-instructions](https://huggingface.co/datasets/AdaptLLM/food-visual-instructions) | [food-VQA-benchmark](https://huggingface.co/datasets/AdaptLLM/food-VQA-benchmark)                   |
| [AdaMLLM-med-8B](https://huggingface.co/AdaptLLM/biomed-LLaVA-NeXT-Llama3-8B) | AdaptLLM/biomed-LLaVA-NeXT-Llama3-8B     | Biomedicine  | open-llava-next-llama3-8b    | [biomed-visual-instructions](https://huggingface.co/datasets/AdaptLLM/biomed-visual-instructions) | [biomed-VQA-benchmark](https://huggingface.co/datasets/AdaptLLM/biomed-VQA-benchmark)                   |
| [AdaMLLM-food-8B](https://huggingface.co/AdaptLLM/food-LLaVA-NeXT-Llama3-8B) |AdaptLLM/food-LLaVA-NeXT-Llama3-8B     | Food  | open-llava-next-llama3-8b    | [food-visual-instructions](https://huggingface.co/datasets/AdaptLLM/food-visual-instructions) |  [food-VQA-benchmark](https://huggingface.co/datasets/AdaptLLM/food-VQA-benchmark)                   |
| [AdaMLLM-med-11B](https://huggingface.co/AdaptLLM/biomed-Llama-3.2-11B-Vision-Instruct) | AdaptLLM/biomed-Llama-3.2-11B-Vision-Instruct     | Biomedicine  | Llama-3.2-11B-Vision-Instruct    | [biomed-visual-instructions](https://huggingface.co/datasets/AdaptLLM/biomed-visual-instructions) | [biomed-VQA-benchmark](https://huggingface.co/datasets/AdaptLLM/biomed-VQA-benchmark)                   |
| [AdaMLLM-food-11B](https://huggingface.co/AdaptLLM/food-Llama-3.2-11B-Vision-Instruct) | AdaptLLM/food-Llama-3.2-11B-Vision-Instruct     | Food | Llama-3.2-11B-Vision-Instruct    | [food-visual-instructions](https://huggingface.co/datasets/AdaptLLM/food-visual-instructions) |  [food-VQA-benchmark](https://huggingface.co/datasets/AdaptLLM/food-VQA-benchmark)                   |


## Setup
We create two separate conda environments.

#### Env 1: To fine-tune the visual instruction synthesizer and post-train LLaVA

1. Clone this repo:
   ```bash
   git clone https://github.com/bigai-ai/QA-Synthesizer.git
   cd QA-Synthesizer
   ```

2. Install the package:
   ```bash
   conda create -n adamllm python=3.10 -y
   conda activate adamllm
   pip install --upgrade pip
   pip install -e .
   ```

3. Install additional packages for training:
   ```bash
   pip install -e ".[train]"
   pip install flash-attn --no-build-isolation
   conda deactivate
   ```

#### Env 2: To synthesize visual instruction tasks and evaluate models on domain-specific tasks
Install vLLM with `pip` or [from source](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source).

  As recommended in the official vLLM documentation, install vLLM in a **fresh new** conda environment:

  ```bash
  conda create -n vllm python=3.10 -y
  conda activate vllm
  pip install vllm  # Ensure vllm>=0.6.2 for compatibility with llama3.2; if llama-3.2 is not used, vllm==0.6.1 is sufficient

  conda deactivate
  ```

## Domain-Specific Visual Instruction Synthesis
The steps in [Synthesis.md](docs/Synthesis.md) reproduce our visual instruction synthesizer and our synthetic data.

## Domain-Specific Single-Stage Post-Training
The steps in [Post-train.md](docs/Post_train.md) reproduce our domain-adapted models. 

## Domain-Specific Task Evaluation

See [Evaluation.md](docs/Evaluation.md) to reproduce our results and evaluate any MLLMs compatible with vLLM.


## License

```text
LICENSE AGREEMENT
Last revision: Sep, 2023
You are granted the right to use the code and/or Database under the following terms, as enlisted in this document (“Beijing Institute for General Artificial Intelligence BIGAI License Agreement”):
·    The code and/or data is strictly for non-commercial academic research only.
·    Any commercial use of the code or data requires prior contact and negotiation of licensing fees with the original authors or Beijing Institute for General Artificial Intelligence (BIGAI).
·    Any new access to the code and/or data shall be established through this form or the official method of distributing the code and/or data. The code and/or data may not be redistributed, in whole or part, or in any format without written prior permission. A reference to  the code and/or data or this License Agreement must be made if you publish information.
·    The code and/or data is provided as is. No liability or responsibility assumed for the authors.
·    The right to revise this License Agreement, in whole or part, at any time without prior notice is reserved by the authors.
·    You warrant that you have the authorization to enter into this License Agreement.
·    You comply with the terms enforced by the corporates whose products were used in collecting the code and/or data. The terms unanimously enforce, including but not limited to, restricting the use of the code and/or data to non-commercial academic research.
```

## Citation
If you find our work helpful, please cite us:

```bibtex
@article{adamllm,
  title={On Domain-Specific Post-Training for Multimodal Large Language Models},
  author={Cheng, Daixuan and Huang, Shaohan and Zhu, Ziyu and Zhang, Xintong and Zhao, Wayne Xin and Luan, Zhongzhi and Dai, Bo and Zhang, Zhenliang},
  journal={arXiv preprint arXiv:2411.19930},
  year={2024}
}
@inproceedings{
adaptllm,
title={Adapting Large Language Models via Reading Comprehension},
author={Daixuan Cheng and Shaohan Huang and Furu Wei},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=y886UXPEZ0}
}
```
