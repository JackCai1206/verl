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
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import Dict, List, Union, Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset that supports both training and validation modes.

    Arguments:
        parquet_files (Union[str, List[str]]): Path(s) to parquet file(s)
        tokenizer: Tokenizer to use for encoding texts
        config: Configuration parameters
        validation_mode (bool): Whether to operate in validation mode (default: False)
            - In training mode: returns concatenated prompt+response with appropriate masks
            - In validation mode: returns only prompt (for generation) and raw response (for evaluation)
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config, validation_mode: bool = False):
        self.validation_mode = validation_mode
        
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 0)  # Default to 0 to enable dynamic padding
        truncation = config.get("truncation", "error")

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.max_length = max_length
        self.use_dynamic_padding = max_length <= 0  # Use dynamic padding if max_length is 0 or negative

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        if len(self.prompt_dict_keys) > 0:
            self.prompts = self.dataframe[self.prompt_key]
            for key in self.prompt_dict_keys:
                # type(x): pandas.core.series.Series
                # type(x[0]): numpy.ndarray
                # type(x[0][0]): dict
                try:
                    self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)
                except Exception:
                    print(f"self.prompts={self.prompts}")
                    raise
        else:
            self.prompts = self.dataframe[self.prompt_key[0]]
        self.prompts = self.prompts.tolist()

        if len(self.response_dict_keys) > 0:
            self.responses = self.dataframe[self.response_key]
            for key in self.response_dict_keys:
                try:
                    self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)
                except Exception:
                    print(f"self.responses={self.responses}")
                    raise
        else:
            self.responses = self.dataframe[self.response_key[0]]
        self.responses = self.responses.tolist()
        
        self.extra_info = self.dataframe.get("extra_info", None)
        
        # Calculate optimal padding length if using dynamic padding
        if self.use_dynamic_padding:
            max_actual_length = 0
            
            # Sample up to 1000 examples for efficiency
            sample_size = min(1000, len(self.prompts))
            sample_indices = torch.randperm(len(self.prompts))[:sample_size].tolist()
            
            for idx in sample_indices:
                prompt = self.prompts[idx]
                response = self.responses[idx]
                
                # Apply chat template
                if isinstance(prompt, str):
                    prompt_chat = [{"role": "user", "content": prompt}]
                else:
                    prompt_chat = prompt
                prompt_chat_str = self.tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
                response_chat_str = response + self.tokenizer.eos_token

                # Tokenize without padding to get actual length
                prompt_ids = self.tokenizer(prompt_chat_str, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
                if self.validation_mode:
                    total_length = prompt_ids
                else:
                    response_ids = self.tokenizer(response_chat_str, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
                    total_length = prompt_ids + response_ids
                max_actual_length = max(max_actual_length, total_length)
            
            # Use the calculated max_length
            print(f"Using dynamic padding with length: {max_actual_length}")
            self.max_length = int(max_actual_length * 1.1) # Add 10% buffer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item) -> Dict[str, Any]:
        """
        Get a dataset item based on the current mode.
        
        In training mode:
            Returns concatenated and padded prompt+response with masks for training
        
        In validation mode:
            Returns just the prompt (left-padded) for generation and raw response for evaluation
        """
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # Apply chat template
        if isinstance(prompt, str):
            prompt_chat = [{"role": "user", "content": prompt}]
        else:
            prompt_chat = prompt
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        
        if self.validation_mode:
            # In validation mode, we only need to tokenize and prepare the prompt
            # We'll also return the raw response text for evaluation
            
            # Tokenize prompt
            prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
            prompt_ids = prompt_ids_output["input_ids"][0]
            prompt_attention_mask = prompt_ids_output["attention_mask"][0]
            
            # Tokenize response
            response_chat_str = response + tokenizer.eos_token
            response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
            response_ids = response_ids_output["input_ids"][0]
            response_length = response_ids.shape[0]
            
            # Left padding for inference
            prompt_length = prompt_ids.shape[0]
            if prompt_length < self.max_length:
                # Left padding (unlike training which uses right padding)
                pad_length = self.max_length - prompt_length
                
                padded_input_ids = torch.ones(size=(pad_length,), dtype=prompt_ids.dtype) * tokenizer.pad_token_id
                padded_attention_mask = torch.zeros(size=(pad_length,), dtype=prompt_attention_mask.dtype)
                
                input_ids = torch.cat((padded_input_ids, prompt_ids))
                attention_mask = torch.cat((padded_attention_mask, prompt_attention_mask))
            elif prompt_length > self.max_length:
                if self.truncation == "left":
                    # Left truncation
                    input_ids = prompt_ids[-self.max_length:]
                    attention_mask = prompt_attention_mask[-self.max_length:]
                elif self.truncation == "right":
                    # Right truncation
                    input_ids = prompt_ids[:self.max_length]
                    attention_mask = prompt_attention_mask[:self.max_length]
                elif self.truncation == "error":
                    raise ValueError(f"{prompt_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")
            else:
                input_ids = prompt_ids
                attention_mask = prompt_attention_mask
            
            # Calculate position IDs based on attention mask
            position_ids = compute_position_id_with_mask(attention_mask)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                # "response_ids": response_ids,
                "response_length": response_length,
                "raw_prompt": prompt_chat_str,
                "raw_response": response_chat_str,  # Pass the ground truth response for evaluation
            }
        else:
            # Training mode - original implementation
            response_chat_str = response + tokenizer.eos_token

            # Tokenize
            prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
            prompt_ids = prompt_ids_output["input_ids"][0]
            prompt_attention_mask = prompt_ids_output["attention_mask"][0]

            response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
            response_ids = response_ids_output["input_ids"][0]
            response_attention_mask = response_ids_output["attention_mask"][0]

            prompt_length = prompt_ids.shape[0]
            response_length = response_ids.shape[0]

            input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
            attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

            # Right padding to max length for training
            sequence_length = input_ids.shape[0]
            if sequence_length < self.max_length:
                padded_input_ids = (
                    torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
                    * tokenizer.pad_token_id
                )
                padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    # Left truncation
                    input_ids = input_ids[-self.max_length:]
                    attention_mask = attention_mask[-self.max_length:]
                elif self.truncation == "right":
                    # Right truncation
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")

            position_ids = compute_position_id_with_mask(attention_mask)

            loss_mask = attention_mask.clone()
            if prompt_length > 1:
                # Mask out prompt for SFT
                loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0
            # Mask out the last token in response
            loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }