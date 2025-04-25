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

import re


def extract_solution(solution_str):
    solution = re.search(r"ANSWER:\s*\n(.*?)(?:\n|$)", solution_str, re.IGNORECASE)
    if solution is None:
        return ''
    final_solution = solution.group(0)
    final_solution = final_solution.split('ANSWER:\n')
    if len(final_solution) < 2:
        return ''
    final_solution = final_solution[1].strip()
    return final_solution


def compute_score(solution_str, ground_truth_str, prompt, method='strict', format_score=0., score=1.):
    solution = extract_solution(solution_str=solution_str)
    ground_truth = extract_solution(solution_str=ground_truth_str)
    if solution is None:
        return 0
    else:
        ans_set = set(map(lambda x: x.strip(), solution.split(',')))
        gt_set = set(map(lambda x: x.strip(), ground_truth.split(',')))
        score = len(ans_set.intersection(gt_set)) / len(gt_set) * score + format_score
        return score