from typing import Optional
from verl.experimental.dynamic_dataset.dynamicgen_dataset import AbstractDataGenerator
from omegaconf import DictConfig
import datasets
from torch.utils.data import Dataset
from random import random
from verl.utils.reward_score import math
from recipe.gas.utils import build_classifier_prompt
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl import DataProto


def build_classifier_prompt(prompt, answer1, answer2):
    gt = 1
    if random() < 0.5:
        answer1, answer2 = answer2, answer1
        gt = 2
    prompt_str = f"""You are given a prompt and two answers. Your job is to classify which of the following answer is correct. Output 1 for the first answer, 2 for the second answer, or 0 if you are not sure. Reason step by step and output the final answer in \\boxed{{}}.
<prompt>
{prompt}
</prompt>
<answer1>
{answer1}
</answer1>
<answer2>
{answer2}
</answer2>
"""
    return [{'content': prompt_str, 'role': 'user'}], gt

class ClassifierDataGenerator(AbstractDataGenerator):
    """
    A noop data gen class that only reappends the first datapoint.
    This class is useful as a placeholder and testing.
    """

    dataset: Optional[Dataset] = None
    def __init__(self, config: DictConfig = None):
        super().__init__(config)

    def build_classifier_batch(self, gen_batch: DataProto, batch: DataProto) -> None:
        prompt_str = gen_batch.non_tensor_batch['raw_prompt']
        gt = [x['ground_truth'] for x in batch.non_tensor_batch['reward_model']]
        ans = [math.remove_boxed(math.last_boxed_only_string(x['response'])) for x in batch.non_tensor_batch['extra_info']]
        messages = []
        classifier_gt = []
        for p, g, a in zip(prompt_str, gt, ans):
            msg, label = build_classifier_prompt(p, g, a)
            messages.extend(msg)
            classifier_gt.append(label)

        self.batch_data = datasets.Dataset.from_dict({
            'prompt': messages,
            'ground_truth': classifier_gt
        })

        # # Prepare the inputs like in rl_dataset.py
        # model_inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
        # # model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        # input_ids = model_inputs.pop("input_ids")
        # attention_mask = model_inputs.pop("attention_mask")
        # breakpoint()
        # input_ids, attention_mask = verl_F.postprocess_data(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_length=self.train_dataset.max_prompt_length,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     left_pad=True,
        #     truncation=self.train_dataset.truncation,
        # )
        # position_ids = compute_position_id_with_mask(attention_mask)

        # classifier_batch = DataProto(
        #     batch={
        #         "input_ids": input_ids,
        #         "attention_mask": attention_mask,
        #         "position_ids": position_ids,
        #     }
        # )
        # return classifier_batch

    def generate(self, dataset: Dataset) -> datasets.Dataset:
        if not self.batch_data:
            raise ValueError("No batch data available to generate from.")
        return self.batch_data
