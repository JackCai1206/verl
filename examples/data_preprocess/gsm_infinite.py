import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

# To extract the solution for each prompts in the dataset
# def extract_solution(solution_str):
# ...

def extract_solution(solution_str):
    solution = re.search("ANSWER:\n(\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('ANSWER:\n')[1].replace(',', '')
    return final_solution

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/staging/zcai75/datasets/igsm')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'InfiniAILab/gsm_infinite_symbolic_0'

    dataset = datasets.interleave_datasets([datasets.load_dataset(data_source, split=f'ops_{n}') for n in range(1, 5)])

    train_dataset, test_dataset = dataset.train_test_split(test_size=0.1).values()
    
    instruction_following = "Let's think step by step and output the final answer after \"####\"."

    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')

            question = question_raw + ' ' + instruction_following

            answer_raw = example.pop('answer')
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
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

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    print(train_dataset[0])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    makedirs(hdfs_dir)

    copy(src=local_dir, dst=hdfs_dir)
