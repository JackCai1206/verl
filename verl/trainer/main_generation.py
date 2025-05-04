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
Generate responses given a dataset of prompts
"""
import re
import ray
import numpy as np
import hydra
import os
from collections import Counter

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils import hdfs_io
from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None

def find_latest_checkpoint(checkpoint_dir, hdfs_dir=None):
    """Find the latest checkpoint based on step number
    
    Args:
        checkpoint_dir: Local directory to search for checkpoints
        hdfs_dir: HDFS directory to search for checkpoints (optional)
        
    Returns:
        Path to the latest checkpoint
    """
    # Function to find latest checkpoint in a directory
    def find_latest_in_dir(directory, use_hdfs=False):
        if use_hdfs:
            if not hdfs_io.exists(directory):
                return None
            checkpoints = hdfs_io.listdir(directory)
        else:
            if not os.path.exists(directory):
                return None
            checkpoints = os.listdir(directory)
            
        steps = []
        for checkpoint in checkpoints:
            if checkpoint.startswith("global_step_"):
                step = extract_step(checkpoint)
                if step is not None:
                    steps.append((step, os.path.join(directory, checkpoint)))
        
        if not steps:
            return None
        
        # Return the path with the highest step
        return sorted(steps, key=lambda x: x[0], reverse=True)[0][1]
    
    # First check local directory
    latest_local = find_latest_in_dir(checkpoint_dir)
    
    # Then check HDFS if configured
    latest_hdfs = None
    if hdfs_dir:
        latest_hdfs = find_latest_in_dir(hdfs_dir, use_hdfs=True)
        if latest_hdfs and (latest_local is None or extract_step(latest_hdfs) > extract_step(latest_local)):
            # Copy from HDFS to local
            local_path = os.path.join(checkpoint_dir, os.path.basename(latest_hdfs))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            hdfs_io.copy(latest_hdfs, local_path)
            return local_path
    
    return latest_local

@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    
    # Check if the model path is a directory that might contain checkpoints
    model_path = config.model.path
    if os.path.isdir(model_path):
        # Try to find the latest checkpoint directly in the specified path
        latest_checkpoint = find_latest_checkpoint(model_path)
        if latest_checkpoint:
            print(f"Found latest checkpoint in model path: {latest_checkpoint}")
            config.model.path = latest_checkpoint
    
    local_path = copy_to_local(config.model.path)
    from verl.utils import hf_tokenizer
    trust_remote_code = config.data.get('trust_remote_code', False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [chat.tolist() for chat in chat_lst]

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    # real_batch_size = data.batch['input_ids'].shape[0]
    config_batch_size = config.data.batch_size
    dispatch_dp_size = wg.world_size
    num_batch = -(-total_samples // config_batch_size)
    output_lst = [[] for _ in range(config.data.n_samples)]
    
    extract_fn = None
    if config.custom_reward_function.path is not None:
        from verl.utils.import_utils import load_extern_type
        extract_fn = load_extern_type(config.custom_reward_function.path, 'extract_solution')

    for batch_idx in range(num_batch):
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
        inputs = tokenizer.apply_chat_template(batch_chat_lst,
                                               add_generation_prompt=True,
                                               padding=True,
                                               truncation=True,
                                               max_length=config.rollout.prompt_length,
                                               return_tensors='pt',
                                               return_dict=True,
                                               tokenize=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

        data = DataProto.from_dict(batch_dict)
        real_batch_size = data.batch['input_ids'].shape[0]
        if real_batch_size % dispatch_dp_size != 0:
            dummy_data_size = dispatch_dp_size - real_batch_size % dispatch_dp_size
            if dummy_data_size <= real_batch_size:
                dummy_data = data[:dummy_data_size]
            else:
                dummy_data = data.repeat(-(-dummy_data_size // real_batch_size))[:dummy_data_size]
            data = DataProto.concat([data, dummy_data])
            print(
                f'real_batch_size {real_batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}, add {dummy_data_size} dummy data'
            )

        batch_size = data.batch['input_ids'].shape[0]
        assert batch_size % dispatch_dp_size == 0, f'batch_size {batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}'

        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
        # START TO GENERATE FOR n_samples TIMES
        for i in range(config.data.n_samples):
            output = wg.generate_sequences(data)
            # remove dummy data
            output = output[:real_batch_size]
            output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                                 skip_special_tokens=False)

            # remove the padding
            pad_token = tokenizer.pad_token
            output_text_unpad = []
            for text in output_text:
                output_text_unpad.append(text.replace(pad_token, ''))

            output_lst[i].extend(output_text_unpad)

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()
    
    if config.data.self_consistency:
        for i, output_batch in enumerate(output_lst):
            if extract_fn is not None: 
                solutions = []
                for output_text in output_batch:
                    sol = extract_fn(output_text)
                    if isinstance(sol, set):
                        sol = sorted(list(sol))
                    solutions.append(sol)
            else:
                solutions = output_batch
            # self-consistency: select the response with most common solution
            # Create counter to find most common solution
            solution_counter = Counter(solutions)
            most_common_solution = solution_counter.most_common(1)[0][0]

            # Find the original response text that produced the most common solution
            most_common_index = solutions.index(most_common_solution)
            most_common_response = output_batch[most_common_index]
            output_lst[i] = [most_common_response]

    # add to the data frame
    dataset['responses'] = output_lst

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)

    return output_text


if __name__ == '__main__':
    main()
