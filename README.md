# CIE



This repository contains the dataset and code of the paper (Under Review):
> **CIE: Controlling Language Model Text Generations Using Continuous Signals**
> <!--
> [[Paper]](https://arxiv.org/pdf/2407.18416) [[arXiv]](https://arxiv.org/abs/2407.18416) [[website]](https://personagym.com)  <br>
> -->

## Setup
```bash
# Environment setup
conda create -n CIE python=3.12 -y
conda activate CIE

# install dependencies
pip install -r requirements.txt
```

## Dataset
While under review, we will not release our training dataset on HuggingFace. However, our training data and all evaluation datasets are present in the `data`.

## Training
We currently provide the exact scripts to train the LLaMA-3-8B-IT, gemma-7B-it, and Qwen1.5-7B-Chat models we experiment with and reproduce our results in `code/scripts`. In order to train on a HuggingFace Instruction tuned model follow the steps below.

Begin by adding the "response template" to line 129 of `verbosity-finetune.py` for the model being added. The response template is a string of tokens present in the model's instruct template right before the "assistant" response. The custom model uses this template to create a label mask during training. 

```bash
if args.model_type == "llama3":
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n"
elif args.model_type == "qwen":
    response_template = "<|im_end|>\n<|im_start|>assistant\n"
elif args.model_type == "gemma":
    tokenizer.add_eos_token = True
    response_template = "\n<start_of_turn>model\n"
```

The only other change within `verbosity-finetune.py` that is needed to be done is adding in the model's isntruct template to the format_instruction function using the same model name as before.

```bash
def format_instruction(model_input):
    """Format instruction based on model type."""
    if args.model_type == "llama3":
        return f'''<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_INSTRUCTION}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{model_input['inputs'].lstrip().rstrip()}<embedding><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{model_input['labels'].lstrip().rstrip()}<|eot_id|>'''
    elif args.model_type == "qwen":
        return f"<|im_start|>system\n{SYSTEM_INSTRUCTION}<|im_end|>\n<|im_start|>user\n{model_input['inputs'].lstrip().rstrip()}<embedding><|im_end|>\n<|im_start|>assistant\n{model_input['labels'].lstrip().rstrip()}<|im_end|>"
    elif args.model_type == "gemma":
        return f"<start_of_turn>user\n{SYSTEM_INSTRUCTION} {model_input['inputs'].lstrip().rstrip()}<embedding><end_of_turn>\n<start_of_turn>model\n{model_input['labels'].lstrip().rstrip()}<end_of_turn>"
```
        
## Evaluation

To start the evaluation of a trained CIE model checkpoint move to the `code` directory and  run the `epoch_evals.py` file. The --checkpoint flag takes in the path to the saved model directory created by HuggingFace Trainer, the --model_name flag takes in the name of the model (ie. llama3, gemma, qwen) to match to the exact instruct template of the trained model, --model_name flag indicates the name to be used when saving results from the given model to be evaluated, and the --benchmarks takes in a list of benchmark names (from validation, validation_ranges, and alpaca-li) to be evaluated on.

An example of running the `epoch_evals.py` file is included below

```bash
python epoch_evals.py --checkpoint /data/group_data/word_count/full/llama3/5e-6_test --model_name llama3 --benchmarks validation
```

<!--
## Bugs or Questions

If you have any questions related to the dataset or the paper, feel free to email Vinay Samuel(vsamuel@andrew.cmu.edu). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation
If you find this repository helpful, please consider citing our paper: 
```bibtex
@article{samuel2024personagym,
  title={PersonaGym: Evaluating Persona Agents and LLMs},
  author={Samuel, Vinay and Zou, Henry Peng and Zhou, Yue and Chaudhari, Shreyas and Kalyan, Ashwin and Rajpurohit, Tanmay and Deshpande, Ameet and Narasimhan, Karthik and Murahari, Vishvak},
  journal={arXiv preprint arXiv:2407.18416},
  year={2024}
}
```
-->

