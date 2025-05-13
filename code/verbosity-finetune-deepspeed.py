import wandb
from datasets import load_dataset 
from dataclasses import dataclass, field
import numpy as np
import os
from tqdm import tqdm 
import torch
from torch.nn import functional as F
import random
import math
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from ControlCausalLM import CausalLMWithControlToken
from transformers import DataCollatorForLanguageModeling
from tokenizers import AddedToken, pre_tokenizers
from transformers import (    
    HfArgumentParser,         
    AutoConfig,
    TrainingArguments, 
    TrainerCallback, 
    Trainer,
    DataCollatorWithPadding,
    AutoTokenizer, 
    AutoModelForCausalLM,
    set_seed
)
from peft import LoraConfig, get_peft_model
from transformers.integrations import WandbCallback
import string
from safetensors.torch import save_model
from transformers import Trainer

@dataclass 
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    model_type: str = field(default=None)
    model_name: str = field(default=None)
    cache_dir: str = field(default='/scratch')
    validation_file: str = field(default=None)
    test_file: str = field(default=None)
    max_source_length: str = field(default=256)
    max_target_length: str = field(default=256)
    max_train_samples: int = field(default=None) 
    max_eval_samples: int = field(default=15)
    wandb_artefact_logging: bool = field(default=False)
    wandb_run_name: str = field(default=None)
    output_prediction_file: str = field(default='predictions.txt')
    previous_checkpoint: str = field(default=None)
    wandb_project_name: str = field(default=None) 


SYSTEM_INSTRUCTION = 'Below is an instruction that is optionally paired with some additional context. Respond appropriately using the context (if any) \n'


def format_test_instruction(prompt):
    """Format a test instruction with the appropriate system prompt and embedding token."""
    return f''' <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_INSTRUCTION}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<embedding><|eot_id|><|start_header_id|>assistant<|end_header_id|>'''


class PeftSavingCallback(TrainerCallback):
    """Callback to save PEFT model during training."""
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


class ComputeCallback(WandbCallback):
    """Callback to log model generations to W&B during training."""
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=64, log_model="checkpoint"): 
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.shuffle(seed=42)
        self.sample_dataset = self.sample_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right" 
        self.max_new_tokens = max_new_tokens
        self.gen_config = AutoConfig.from_pretrained(trainer.model.name_or_path)

    def generate(self, prompt, max_new_tokens):
        """Generate text from a prompt using the model."""
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=tokenized_prompt.input_ids.cuda(), 
                attention_mask=tokenized_prompt.attention_mask.cuda(), 
                max_new_tokens=max_new_tokens, 
                pad_token_id=self.tokenizer.eos_token_id
            )
            predictions = self.tokenizer.decode(output[0][len(prompt[0]):], skip_special_tokens=True)
        return predictions 
    
    def samples_table(self, examples, max_new_tokens):
        """Create a W&B table with sample generations."""
        records_table = wandb.Table(columns=["prompt", "generation"])
        for example in tqdm(examples, leave=False):
            prompt = example["conversations"][0]
            prompt = format_test_instruction(prompt)
            generation = self.generate(prompt=prompt, max_new_tokens=max_new_tokens)
            records_table.add_data(prompt, generation)
        return records_table
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Log sample predictions to W&B during evaluation."""
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset, self.max_new_tokens)
        self._wandb.log({"sample_predictions": records_table})

    
def main():
    """Main training function."""
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()    
    wandb.init(
        project="verbosity-control", 
        entity="ltfighter15", 
        name=args.wandb_run_name, 
        group=f'{args.wandb_run_name}_ddp'
    )    
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.model_type == "mistral":
        tokenizer.add_eos_token = True
        response_template = "[/INST]"
    elif args.model_type == "llama3":
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n"
    elif args.model_type == "qwen":
        response_template = "<|im_end|>\n<|im_start|>assistant\n"
    elif args.model_type == "gemma":
        tokenizer.add_eos_token = True
        response_template = "\n<start_of_turn>model\n"
    
    response_ids = tokenizer(response_template, return_tensors="pt").input_ids
    if args.model_type == "mistral" or args.model_type == "gemma":
        response_ids = response_ids[:, :-1]

    tokenizer.add_tokens(["<embedding>"])
    control_id = tokenizer.convert_tokens_to_ids("<embedding>")

    # Prepping Data 
    train_dataset = None #TODO 
    test_dataset = load_dataset('GAIR/lima')['test']
        
    def format_instruction(model_input):
        """Format instruction based on model type."""
        if args.model_type == 'llama2': 
            return [f"""<s> [INST] <<SYS>> {SYSTEM_INSTRUCTION} <</SYS>>{model_input.lstrip().rstrip()} [/INST]"""]
        elif args.model_type == "llama3":
            return f'''<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_INSTRUCTION}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{model_input['inputs'].lstrip().rstrip()}<embedding><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{model_input['labels'].lstrip().rstrip()}<|eot_id|>'''
        elif args.model_type == "mistral":
            return f'''<s>[INST] {SYSTEM_INSTRUCTION} {model_input['inputs'].lstrip().rstrip()}<embedding> [/INST]{model_input['labels'].lstrip().rstrip()}</s>'''
        elif args.model_type == "qwen":
            return f"<|im_start|>system\n{SYSTEM_INSTRUCTION}<|im_end|>\n<|im_start|>user\n{model_input['inputs'].lstrip().rstrip()}<embedding><|im_end|>\n<|im_start|>assistant\n{model_input['labels'].lstrip().rstrip()}<|im_end|>"
        elif args.model_type == "gemma":
            return f"<start_of_turn>user\n{SYSTEM_INSTRUCTION} {model_input['inputs'].lstrip().rstrip()}<embedding><end_of_turn>\n<start_of_turn>model\n{model_input['labels'].lstrip().rstrip()}<end_of_turn>"

    def preprocess_train_function(examples):
        """Preprocess training examples."""
        model_inputs = {}
        inputs = [ex for ex in examples['instruction']]  
        targets = [ex for ex in examples['answer']]
        model_inputs['inputs'] = inputs
        model_inputs['labels'] = targets
        return model_inputs
    
    def preprocess_test_function(examples):
        """Preprocess test examples."""
        model_inputs = {}
        test_inputs = [ex[0] for ex in examples['conversations']]
        model_inputs['inputs'] = test_inputs
        return model_inputs
    
    if training_args.do_train:
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
        if 'random' in training_args.output_dir:
            generated_seed = random.choice(range(args.max_train_samples))
            train_dataset = train_dataset.shuffle(seed=generated_seed)
            print(f'Have reshuffled dataset for random sampling with {generated_seed}...')
            train_dataset = train_dataset.select(range(args.max_train_samples))
            
        train_dataset = train_dataset.map(
            preprocess_train_function,
            batched=True,
            desc=f"Creating input and target training samples with budget {args.max_train_samples}",
        )
        
    if training_args.do_eval:
        if args.max_eval_samples is not None:
            eval_dataset = eval_dataset.map(
                preprocess_train_function,
                batched=True,
                desc="Creating input and target validation samples using the Training Dataset!.",
            )
            test_dataset = test_dataset.select(range(args.max_eval_samples))
            test_dataset = test_dataset.map(
                preprocess_test_function,
                batched=True,
                desc="Sampling LIMA Test Prompts for Checking Inference!",
            )
    
    train_dataset = train_dataset.remove_columns(['instruction', 'answer', "dataset"])
    model = CausalLMWithControlToken(model_name=args.model_name, response_ids=response_ids)

    class CustomDataCollator(DataCollatorWithPadding):
        """Custom data collator that handles control tokens."""
        def __init__(self, tokenizer):
            super().__init__(tokenizer)

        def __call__(self, examples):
            ans_lens = [ex["ans_len"] for ex in examples]
            
            inputs = [format_instruction(ex) for ex in examples]
            batch = tokenizer(inputs, padding=True, return_tensors="pt")
            ctrl_mask = torch.zeros_like(batch["input_ids"])

            for i in range(batch["input_ids"].shape[0]):
                ctrl_mask[i][batch["input_ids"][i] == control_id] = ans_lens[i]
            
            batch["verbosity_mask"] = ctrl_mask
            return batch
    
    data_collator = CustomDataCollator(tokenizer=tokenizer)

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    set_seed(training_args.seed)
    training_args.ddp_find_unused_parameters = False
    training_args.remove_unused_columns = False

    trainer = Trainer(
        model=model, 
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer 
    )

    if args.previous_checkpoint is not None:
        print('Found a previous checkpoint - so resuming training from there...')
        train_result = trainer.train(resume_from_checkpoint=args.previous_checkpoint)
    else: 
        train_result = trainer.train()
    
    trainer.save_model()
    wandb.finish()


if __name__ == "__main__":
    main()
