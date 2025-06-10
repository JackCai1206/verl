# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the AIME datasets to parquet format
"""

import argparse
from functools import partial
import os
import yaml

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=os.path.join(os.path.expanduser("~"), "Jack/data/verl"))
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--datasets_config", default="recipe/reasoning_transfer/datasets_config.yaml", type=str,
                       help="JSON string or file path with dataset configurations")

    args = parser.parse_args()

    if os.path.isfile(args.datasets_config):
        with open(args.datasets_config, 'r') as f:
            if args.datasets_config.endswith('.yaml') or args.datasets_config.endswith('.yml'):
                datasets_config = yaml.safe_load(f)

    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}.\n"

    def make_map_fn(dataset_name, question_col, answer_col, extra_cols=None):
        def process_fn(example, idx):
            question = example[question_col] + " " + instruction_following
            if answer_col in example:
                answer = str(example[answer_col])
            else:
                answer = ""
            
            extra_info = {"index": idx}
            if extra_cols:
                for new_key, orig_col in extra_cols.items():
                    if orig_col in example:
                        extra_info[new_key] = example[orig_col]
            
            example = {
                "data_source": dataset_name,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": extra_info,
            }
            return example

        def process_fn_sudoku(example, idx):
            # Convert flattened list to indexed format
            puzzle_flat = example[question_col]
            
            # Create indexed list format
            grid_str = "Sudoku Puzzle (81 cells, rows 1-9, columns A-I):\n"
            for i in range(81):
                row = (i // 9) + 1
                col = chr(ord('A') + (i % 9))
                cell_value = '_' if puzzle_flat[i] == 0 else str(puzzle_flat[i])
                grid_str += f"{col}{row}: {cell_value}\n"
            
            question = f"{grid_str}\n" + \
                      "Solve this Sudoku puzzle step by step. Fill in the empty cells (marked with '_') with digits 1-9 such that:\n" + \
                      "- Each row contains all digits 1-9 exactly once\n" + \
                      "- Each column contains all digits 1-9 exactly once\n" + \
                      "- Each 3x3 box contains all digits 1-9 exactly once\n\n" + \
                      "You can reference cells by their position (e.g., A1, B2, I9). " + \
                      "Provide your reasoning step by step, then put your final answer as a list of 81 numbers (reading left-to-right, top-to-bottom) within \\boxed{}."
            
            answer = str(example[answer_col])

            extra_info = {"index": idx}
            if extra_cols:
                for new_key, orig_col in extra_cols.items():
                    if orig_col in example:
                        extra_info[new_key] = example[orig_col]
            
            example = {
                "data_source": dataset_name + '_clue_' + str(example['clue_numbers']),
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "sudoku", "ground_truth": answer},
                "extra_info": extra_info,
            }
            return example

        return process_fn if dataset_name != "jackcai1206/sudoku_easy2hard" else process_fn_sudoku

    for config in datasets_config:
        dataset_path = config["path"]
        question_col = config["question_column"]
        answer_col = config.get("answer_column", "")
        extra_cols = config.get("extra_columns", {})
        splits = config.get("split", None)

        dataset_dir = os.path.join(args.local_dir, dataset_path.replace("/", "_"))
        
        # # Skip if dataset already exists
        # if os.path.exists(dataset_dir) and all(
        #     os.path.exists(os.path.join(dataset_dir, f"{split}.parquet")) for split in (splits if isinstance(splits, list) else [splits])
        # ):
        #     print(f"Dataset {dataset_path} already exists at {dataset_dir}, skipping...", flush=True)
        #     continue
        
        print(f"Loading {dataset_path}...", flush=True)
        full_dataset = datasets.load_dataset(dataset_path, trust_remote_code=True)
        
        os.makedirs(dataset_dir, exist_ok=True)
        
        if splits is None:
            splits_to_process = full_dataset.keys()
        else:
            splits_to_process = splits if isinstance(splits, list) else [splits]
        
        if dataset_path == "jackcai1206/sudoku_easy2hard":
            num_classes = 81-17
            full_dataset['train'] = full_dataset['train'].cast_column(
                'clue_numbers', datasets.ClassLabel(num_classes=num_classes, names=[str(i) for i in range(17, 81)])
            )
            full_dataset = full_dataset['train'].train_test_split(
                train_size=num_classes * 1000,
                test_size=num_classes * 5,
                stratify_by_column='clue_numbers',
                shuffle=True
            )
            splits_to_process = ['train', 'test']

        for split_name in splits_to_process:
            print(f"Processing split: {split_name}", flush=True)
            dataset = full_dataset[split_name]
            dataset = dataset.map(function=make_map_fn(dataset_path, question_col, answer_col, extra_cols), 
                                 with_indices=True, remove_columns=[question_col, answer_col] + list(extra_cols.values()), num_proc=16)
            dataset.to_parquet(os.path.join(dataset_dir, f"{split_name}.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)

