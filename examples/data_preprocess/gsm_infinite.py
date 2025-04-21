import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

# To extract the solution for each prompts in the dataset
# def extract_solution(solution_str):
# ...

def extract_solution(solution_str):
    solution = re.search(r"ANSWER:\s*\n(.*?)(?:\n|$)", solution_str, re.IGNORECASE)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('ANSWER:\n')[1].replace(',', '').strip()
    return final_solution

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/Jack/datasets/igsm')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'jackcai1206/gsm_infinite_symbolic_0'

    ds = datasets.load_dataset(data_source)
    
    for round in range(1, 5):
        dataset = datasets.interleave_datasets([ds[f'ops_{n}'] for n in range(round, 5 + round)], seed=42)

        train_dataset, test_dataset = dataset.train_test_split(test_size=0.1).values()

        def make_map_fn(split):

            def process_fn(example, idx):
                solution = extract_solution(example['solution'])
                full_question = example['messages'][0]['content']
                data = {
                    "data_source": data_source,
                    "prompt": example.pop('messages'),
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                        'full_question': full_question,
                        'full_solution': example['solution']
                    },
                }
                return data

            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
        
        print(train_dataset[0])

        local_dir = args.local_dir
        hdfs_dir = args.hdfs_dir

        train_dataset.to_parquet(os.path.join(local_dir, f'train_{round}.parquet'))
        test_dataset.to_parquet(os.path.join(local_dir, f'test_{round}.parquet'))

        if hdfs_dir is not None:    
            makedirs(hdfs_dir)
            copy(src=local_dir, dst=hdfs_dir)
