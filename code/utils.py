import json
import os 

# Default paths that can be overridden via environment variables
DEFAULT_INFERENCE_PATH = 'results/'
inference_path = os.environ.get('INFERENCE_PATH', DEFAULT_INFERENCE_PATH)
SYSTEM_PROMPT = 'Below is an instruction that is optionally paired with some additional context. Write a response that appropriately follows the instruction using the context (if any) '

def dump_predictions(predictions, file):
    with open(os.path.join(inference_path, f'{file}.jsonl'), 'a') as write_file:
        for prediction in predictions:
            write_file.write(json.dumps({'prediction':prediction}, ensure_ascii = False))
            write_file.write('\n')

