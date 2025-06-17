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

# Import killer sudoku parsing function
try:
    from recipe.reasoning_transfer.killer_sudoku import parse_puzzle_string
except ImportError:
    # Fallback if import fails
    def parse_puzzle_string(puzzle_str):
        from collections import defaultdict
        lines = puzzle_str.strip().splitlines()
        if len(lines) != 2 or len(lines[0]) != 81:
            raise ValueError("Invalid puzzle string format")
        seq = lines[0]
        cage_id_map = [['']*9 for _ in range(9)]
        for idx, ch in enumerate(seq):
            r, c = divmod(idx, 9)
            cage_id_map[r][c] = ch
        
        cage_cells = defaultdict(list)
        for r in range(9):
            for c in range(9):
                letter = cage_id_map[r][c]
                cage_cells[letter].append((r,c))
        
        cage_sums = {}
        for token in lines[1].split(';'):
            if not token:
                continue
            letter, sval = token.split(':')
            cage_sums[letter] = int(sval)
        
        return cage_id_map, cage_cells, cage_sums


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
            
            # Create 9x9 grid format
            grid_str_lines = ["Sudoku Puzzle:"] # Title
            for r in range(9):
                row_str_parts = []
                for c in range(9):
                    cell_value = '_' if str(puzzle_flat[r * 9 + c]) == '0' else str(puzzle_flat[r * 9 + c])
                    row_str_parts.append(cell_value)
                grid_str_lines.append(" ".join(row_str_parts))
            
            # Correcting to use actual newlines for the question string construction
            grid_str = "\n".join(grid_str_lines) 

            question = f"{grid_str}\n" + \
                      "Solve this Sudoku puzzle step by step. Fill in the empty cells (marked with '_') with digits 1-9 such that:\n" + \
                      "- Each row contains all digits 1-9 exactly once\n" + \
                      "- Each column contains all digits 1-9 exactly once\n" + \
                      "- Each 3x3 box contains all digits 1-9 exactly once\n\n" + \
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

        def process_fn_killer_sudoku(example, idx):
            # Parse the puzzle string format from killer sudoku
            puzzle_str = example[question_col]
            cage_id_map, cage_cells, cage_sums = parse_puzzle_string(puzzle_str)

            # Create 9x9 grid showing cage structure
            grid_str_lines = ["Killer Sudoku Puzzle:"]
            for r in range(9):
                row_str_parts = []
                for c in range(9):
                    cage_id = cage_id_map[r][c]
                    row_str_parts.append(cage_id)
                grid_str_lines.append(" ".join(row_str_parts))
            
            grid_str = "\n".join(grid_str_lines)
            
            # Format cage sums for display
            cage_sums_str = "Cage sums:\n"
            for cage_id in sorted(cage_sums.keys()):
                cage_sums_str += f"Cage {cage_id}: {cage_sums[cage_id]}\n"

            question = f"{grid_str}\n\n{cage_sums_str}\n" + \
                      "Solve this Killer Sudoku puzzle step by step. Fill in all cells with digits 1-9 such that:\n" + \
                      "- Each row contains all digits 1-9 exactly once\n" + \
                      "- Each column contains all digits 1-9 exactly once\n" + \
                      "- Each 3x3 box contains all digits 1-9 exactly once\n" + \
                      "- Each cage (group of cells marked with the same letter) sums to the specified total\n" + \
                      "- No digit repeats within a cage\n\n" + \
                      "Provide your reasoning step by step, then put your final answer as a list of 81 numbers (reading left-to-right, top-to-bottom) within \\boxed{}."

            # Convert solution from 9x9 grid to flat list
            solution_grid = example[answer_col]
            if isinstance(solution_grid[0], list):
                # If it's a 9x9 grid, flatten it
                answer = str([cell for row in solution_grid for cell in row])
            else:
                # If it's already flat, use as is
                answer = str(solution_grid)

            extra_info = {"index": idx}
            if extra_cols:
                for new_key, orig_col in extra_cols.items():
                    if orig_col in example:
                        extra_info[new_key] = example[orig_col]
            
            example = {
                "data_source": dataset_name + '_diff_' + str(example.get('difficulty', 'unknown')),
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "killer_sudoku", "ground_truth": answer},
                "extra_info": extra_info,
            }
            return example

        if dataset_name == "jackcai1206/sudoku_easy2hard":
            return process_fn_sudoku
        elif dataset_name == "jackcai1206/killer-sudoku-puzzles":
            return process_fn_killer_sudoku
        else:
            return process_fn

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

