import os
import subprocess
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default='/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf', 
                        help='path of the model')
    parser.add_argument("--model_name", type=str, default='llama3', 
                        help='name of the model for file organization')
    parser.add_argument("--benchmarks", nargs='+', default=["validation"],
                        help='list of benchmarks to evaluate')
    args = parser.parse_args()
    
    checkpoint_folders = [args.checkpoint]
    benchmarks = args.benchmarks
    
    for checkpoint_folder in checkpoint_folders:
        path_parts = checkpoint_folder.split("/")
        lr = path_parts[-1]
        model_name = args.model_name
        
        bs = 32
        
        if args.model_name == "mistral":
            model = "/data/models/huggingface/mistralai/Mistral-7B-Instruct-v0.3"
        elif args.model_name == "qwen":
            model = "/data/models/huggingface/qwen/Qwen1.5-7B"
        elif args.model_name == "gemma":
            model = "/data/group_data/models--google--gemma-7b-it/snapshots/9c5798d27f588501ce1e108079d2a19e4c3a2353"
            bs = 8
        elif args.model_name == "llama3":
            model = "/data/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct"
        
        # Get all checkpoints in the folder
        checkpoints = sorted([cp for cp in os.listdir(checkpoint_folder) if cp.startswith("checkpoint")])
        
        for benchmark in benchmarks:
            for checkpoint in checkpoints:
                checkpoint_num = checkpoint.split('-')[-1]
                
                prediction_dir = f"full_finetune/word_count/{model_name}/{benchmark}"
                full_pred_dir = f'../results/{prediction_dir}'
                os.makedirs(full_pred_dir, exist_ok=True)
                
                prediction_file_path = f"{prediction_dir}/{checkpoint_num}"
                
                # Construct the command with the updated prediction file path
                base_command = (
                    f"python eval.py "
                    f"--model_identifier {model_name} "
                    f"--model {model} "
                    f"--bs {bs} "
                    f"--prediction_file_name {prediction_file_path} "
                    f"--model_saved_dir {checkpoint_folder}/{checkpoint} "
                    f"--benchmark {benchmark}"
                )
                
                print(f"Running: {base_command}")
                subprocess.run(base_command, shell=True, check=True)