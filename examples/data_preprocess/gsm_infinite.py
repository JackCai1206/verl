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
    final_solution = final_solution.split('ANSWER:\n')[1].strip()
    return final_solution

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/Jack/datasets/igsm')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'jackcai1206/gsm_infinite_symbolic_0'

    ds = datasets.load_dataset(data_source)
    
    train_datasets = []
    test_datasets = []
    for n in range(1, 12):
        train_dataset, test_dataset = ds[f'ops_{n}'].train_test_split(test_size=100).values()
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    for round in range(1, 5):
        train_dataset = datasets.interleave_datasets(train_datasets[round-1:round + 4], seed=42)
        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        print(train_dataset[0])

        train_dataset.to_parquet(os.path.join(args.local_dir, f'train_{round}.parquet'))

        if args.hdfs_dir is not None:    
            makedirs(args.hdfs_dir)
            copy(src=args.local_dir, dst=args.hdfs_dir)

    for n in range(1, 12):
        test_dataset = test_datasets[round - 1]
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

        test_dataset.to_parquet(os.path.join(args.local_dir, f'test_op_{n}.parquet'))
        breakpoint()

        if args.hdfs_dir is not None:
            makedirs(args.hdfs_dir)
            copy(src=args.local_dir, dst=args.hdfs_dir)
