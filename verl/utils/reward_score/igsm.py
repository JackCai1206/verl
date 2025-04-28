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
        return None
    final_solution = solution.group(0)
    final_solution = final_solution.split('ANSWER:\n')
    if len(final_solution) < 2:
        return None
    final_solution = final_solution[1].strip()
    ans_set = set(map(lambda x: x.strip(), final_solution.split(',')))
    return ans_set


def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    ans_set = extract_solution(solution_str=solution_str)
    if ans_set is None:
        return 0
    else:
        score = len(ans_set.intersection(ground_truth)) / len(ground_truth) * score + format_score
        return score