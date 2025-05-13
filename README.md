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
While under review, we will not be releasing our training dataset on HuggingFace. However our training data and all out evalaution datasets are present in the `data`.

## Training


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

