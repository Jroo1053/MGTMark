"""
LLM Artifact Detection and Evasion Tester.

Copyright (C) 2024 Joseph Frary

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

"""
TF_UTILS - All tensorflow utilities are kept here.
Kept separate to avoid loading tensorflow in non tf test
"""

import tensorflow as tf
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

# DTYPE = torch.bfloat16
# GEMMA_TOKENIZER = AutoTokenizer.from_pretrained("google/gemma-2b-it")
# GEMMA_MODEL = AutoModelForCausalLM.from_pretrained(
#     "google/gemma-2b-it",
#     device_map="cuda",
#     torch_dtype=DTYPE)
#
#
# def gemma_rewrite(subject,base_prompt, is_tonal=False):
#     gemma_chat = [
#         {
#             "role": "user",
#             "content": f"Write a wiki intro for the following subject {subject}"
#         }
#     ]
#     prompt = GEMMA_TOKENIZER.apply_chat_template(
#         gemma_chat, tokenize=False, add_generation_prompt=False
#     )
#     inputs = GEMMA_TOKENIZER.encode(
#         prompt, add_special_tokens=False, return_tensors="pt"
#     )
#     outputs = GEMMA_MODEL.generate(
#         input_ids=inputs.to(GEMMA_MODEL.device),
#         max_new_tokens=256
#     )
#     return GEMMA_TOKENIZER.decode(outputs[0]).split("<end_of_turn>")[1]
#
#
# def gemma_rewrite_mappable(entry, args, label):
#     result = gemma_rewrite(entry[label])
#     return {
#         "generated_intro": result
#     }


def cuda_check():
    """
    Basic utility to check if tf and CUDA are working together.
    :return: True if CUDA is up and running.
    """
    if not len(tf.config.list_physical_devices(
            'GPU')):
        return False
    return True


