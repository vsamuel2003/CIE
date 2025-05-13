import argparse
from utils import SYSTEM_PROMPT, dump_predictions
from datasets import load_dataset
from dataclasses import dataclass, field
import numpy as np
import os
import tqdm
import json
import torch
# Keep safetensors import for when it's available
from safetensors.torch import load_file
from ControlCausalLM import CausalLMWithControlToken
from tokenizers import AddedToken, pre_tokenizers
from transformers import (    
    AutoTokenizer, 
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
from transformers.integrations import WandbCallback
from accelerate import PartialState
from transformers.utils import logging
logging.get_logger("transformers").setLevel(logging.ERROR)

def model_formatter(test_example, model_type):
    if model_type == 'llama':
        return f"""{SYSTEM_PROMPT}\n ### Instruction: {test_example} \n"""
    elif model_type == 'llama2':
        return f"""<s> [INST] <<SYS>> {SYSTEM_PROMPT} <</SYS>>{test_example} [/INST]"""
    elif model_type == 'llama3':
        return f"""<|start_header_id|>system<|end_header_id|>{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{test_example}<embedding><|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    elif model_type == "mistral":
        return f'''<s>[INST] {SYSTEM_PROMPT} {test_example.lstrip().rstrip()}<embedding> [/INST]'''
    elif model_type == "qwen":
        return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{test_example.lstrip().rstrip()}<embedding><|im_end|>\n<|im_start|>assistant\n"
    elif model_type == "gemma":
        return f"<start_of_turn>user\n{SYSTEM_PROMPT} {test_example.lstrip().rstrip()}<embedding><end_of_turn>\n<start_of_turn>model\n"


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def load_jsonl(filepath):
    data_list = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data_list.append(json.loads(line))
    return data_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf', help='path of the model')
    parser.add_argument("--model_identifier", type=str, default='llama2', help='llama or llama2 depending on the base model for the model being evaluated. Used to appropriate prompt formatting.')
    parser.add_argument("--model_saved_dir", type=str, default='finetune-10k', help='path to folder where finetuned model weights are saved')
    parser.add_argument("--bs", type=int, default=None) # bs greater than 1 is not supported yet - needs batch_decode
    parser.add_argument("--summary_results_dir", type=str, default=None)
    parser.add_argument("--prediction_file_name", type=str, default='llama2_lima')
    parser.add_argument("--benchmark", type=str, default='alpaca')

    
    def process_test(examples):
        ans_len = [ex["ans_len"] if "ans_len" in ex else ex["max_len"] for ex in examples]
        inputs = [model_formatter(ex["instruction"], args.model_identifier) for ex in examples]

        model_input = tokenizer(inputs, return_tensors='pt', truncation=True, padding='max_length', max_length=256).to(model.device)
        ctrl_mask = torch.zeros_like(model_input["input_ids"])

        for i in range(model_input["input_ids"].shape[0]):
            ctrl_mask[i][model_input["input_ids"][i] == control_id] = ans_len[i]
        
        model_input["verbosity_mask"] = ctrl_mask.to(model.device)

        return model_input

    def get_embeddings(input_ids, verbosity, model):
        ctrl_mask = verbosity.clamp(0, 1).bool()
        vocab_size = model.config.vocab_size
        input_ids = torch.where(ctrl_mask, torch.full_like(input_ids, vocab_size - 1), input_ids)
        ctrl_mask = ctrl_mask.unsqueeze(-1)

        ctrl_normed = (
            (verbosity).clamp(model.ctrl_min, model.ctrl_max)
            - model.ctrl_min
        ) / (model.ctrl_max - model.ctrl_min)

        control_values = torch.stack((ctrl_normed, 1 - ctrl_normed), dim=2).to(
            model.control_embeds.weight.dtype
        )

        control_embeds = control_values.to(model.device) @ model.control_embeds.weight

        token_embeds = model.get_input_embeddings()(input_ids).to(torch.bfloat16)
        inputs_embeds = torch.where(ctrl_mask, control_embeds, token_embeds)
        inputs_embeds = inputs_embeds.bfloat16()

        return inputs_embeds.to(model.device)

    args = parser.parse_args() 

    if args.model_identifier is None: 
        model_identifier = '-'.join(args.model.strip().split('/')[-4:-1]).replace('/','-')
    else: 
        model_identifier = args.model_identifier
    print(f'Model Identifier: {model_identifier}')

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.model_identifier == "mistral":
        tokenizer.add_eos_token = True
        response_template = "[/INST]"
    elif args.model_identifier == "llama3":
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n"
    elif args.model_identifier == "qwen":
        response_template = "<|im_end|>\n<|im_start|>assistant\n"
    elif args.model_identifier == "gemma":
        response_template = "\n<start_of_turn>model\n"

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.add_tokens(["<embedding>"])
    control_id = tokenizer.convert_tokens_to_ids("<embedding>")

    response_ids = tokenizer(response_template, return_tensors="pt").input_ids

    if args.model_identifier == 'llama2': 
        tokenizer.pad_token = "[PAD]" 

    if args.model_identifier == 'llama3':
        tokenizer.pad_token = "<|end_of_text|>"

    device_string = PartialState().process_index
    model = CausalLMWithControlToken(model_name=args.model, response_ids=response_ids)
    
    # Check if safetensors file exists, otherwise try PyTorch bin
    safetensors_path = f"{args.model_saved_dir}/model.safetensors"
    pytorch_path = f"{args.model_saved_dir}/pytorch_model.bin"
    
    if os.path.exists(safetensors_path):
        print(f"Loading model from SafeTensors: {safetensors_path}")
        state_dict = load_file(safetensors_path, device="cpu")
    elif os.path.exists(pytorch_path):
        print(f"Loading model from PyTorch checkpoint: {pytorch_path}")
        state_dict = torch.load(pytorch_path, map_location="cpu")
    
    model.load_state_dict(state_dict)
    model.to("cuda")

    print(f'Max control is {model.ctrl_max}')
    print(f'Benchmark is {args.benchmark}')

    file_path = f'../data/{args.benchmark}.jsonl'
    test_inputs = load_jsonl(file_path)

    all_preds = []
    print('Starting Inference ... ')    

    for idx in tqdm.tqdm(range(0, len(test_inputs), args.bs)):
        local_batch_test_inputs = test_inputs[idx: idx + args.bs]
        local_batch = process_test(local_batch_test_inputs)

        with torch.no_grad():
            terminators = [
                tokenizer.eos_token_id,
            ]

            input_ids = local_batch.input_ids
            input_embeds = get_embeddings(input_ids, local_batch["verbosity_mask"], model)

            outputs = model.generate(
                input_ids.cuda(), 
                inputs_embeds=input_embeds,
                attention_mask=local_batch.attention_mask.cuda(),
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            response = outputs[:, input_ids.shape[-1]:]
            predictions = tokenizer.batch_decode(response, skip_special_tokens=True)
            all_preds.extend(predictions)

    dump_predictions(all_preds, args.prediction_file_name)