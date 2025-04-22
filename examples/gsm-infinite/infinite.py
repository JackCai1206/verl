from datasets import DatasetDict, concatenate_datasets, load_dataset
import numpy as np
import re, argparse, os
from functools import lru_cache, partial, cache
import pprint

examples_per_step = 5_000
test_examples_per_step = int(.15 * examples_per_step)
stride = 3
window_size = 5
key_prefix = "ops_"
n_levels = 5
data_source = "sbaumohl/gsm_infinite_symbolic_2.5k_10"

pattern = r"ANSWER:\s*(V\d+(?:\s*,\s*V\d+)*)\.?\s*\Z"

# regex is expensive, let's cache this
@lru_cache(maxsize=1024)
def extract_solution_set(s: str):
    match = re.search(pattern, s)
    if match is None:
        return None
    return set(map(lambda x: x.strip(),match.group(1).split(',')))


# reward function scoring
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    ground = set(ground_truth.split(',')) 
    solution = extract_solution_set(solution_str)
    if solution is None:
        return 0
    return 1 if ground == solution else 0

# add a row to each data item that represents a unique id
def process_fn(split, example, idx):
    question_raw = example.pop('question')
    answer_raw = example.pop('solution')
    try:
        solution = ','.join(list(extract_solution_set(answer_raw)))
    except:
        print("Issue with: ", answer_raw)
        exit(1)
    data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": example["messages"][0]["content"],
        }],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx,
            'answer': answer_raw,
            "question": question_raw,
        }
    }
    return data

test_map_fn = partial(process_fn, "test")
train_map_fn = partial(process_fn, "train")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./gsm-infinite/')
    args = parser.parse_args()

    dataset = load_dataset(data_source)

    output_dataset = DatasetDict()
    num_per_grab = int((1 / window_size) * (test_examples_per_step + examples_per_step))

    for i, s in enumerate(range(1, n_levels * stride + 1, stride)):
        l = list(range(s, s + window_size))
        samples = []
        for x in l:
            k = f'{key_prefix}{x}'
            indxs = np.random.choice(len(dataset[k]), size=num_per_grab, replace=False)
            samples.append(dataset[k].select(indxs))
            
        level_data = concatenate_datasets(samples)
        split_data = level_data.train_test_split(test_size=test_examples_per_step, shuffle=True)

        # Now map for prep to use with verl
        split_data["train"] = split_data["train"].map(function=train_map_fn, with_indices=True)
        split_data["test"] = split_data["test"].map(function=test_map_fn, with_indices=True)

        split_data["train"].to_parquet(os.path.join(args.local_dir, f'level_{i + 1}', 'train.parquet'))
        split_data["test"].to_parquet(os.path.join(args.local_dir, f'level_{i + 1}', 'test.parquet'))

