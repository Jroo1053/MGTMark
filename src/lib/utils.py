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
import secrets

import openai
import requests

"""
Utils, various basic functions - non tensorflow.
"""

import random
import math
from collections import Counter
import re
from openai import OpenAI
from bert_score import BERTScorer
import json
from transformers import pipeline
from datasets import Dataset
from line_profiler_pycharm import profile

ORIG_CHECK_URL = "https://api.originality.ai/api/v1/account/credits/balance"
WINSTON_CHECK_URL = "https://api.gowinston.ai/functions/v1/predict"
SPLITTER = "-" * 40


"""
Define attack functions and names so that they can be called dynamically.
funcs must be mappable i.e take a dict return a dict otherwise they cannot 
be applied via map()
"""



def get_text_similarity(base_entries: list, new_entries: list) -> list:
    scorer = BERTScorer(lang="en")
    precsion, recall, f1 = scorer.score(
        base_entries, new_entries
    )
    return f1


def check_for_dataset(dataset_name:str, methods:str):
    return True

def load_auth_creds(file: str, model: str) -> str:
    with open(file, "r") as cred_file:
        key = cred_file.readline()
    if model == "GPT3" and gpt_auth_check(key):
        return key
    elif model == "GPT3":
        raise ValueError("Failed to authorise with OpenAI API")
    if model == "ORIG" and orig_auth_check(key):
        return key
    elif model == "ORIG":
        raise ValueError("Failed to authorise with Originality.ai API")
    if model == "WINSTON" and winston_auth_check(key):
        return key
    elif model == "WINSTON":
        raise ValueError("Failed to authorise with Winston API")


def winston_auth_check(key: str) -> bool:
    headers = {
        "Authorization": f"Bearer {key} ",
        "Content-Type": "application/json"
    }
    payload = {
        "language": "en",
        "sentences": True,
        "text": secrets.token_urlsafe(16),
        "version": "3.0"
    }
    response = requests.request(
        "POST", url=WINSTON_CHECK_URL, json=payload, headers=headers
    )
    if response.status_code == 403:
        return True
    return False


def orig_auth_check(key: str) -> bool:
    headers = {
        "X-OAI-API-KEY": key,
        "Accept": "application/json"
    }
    response = requests.request(
        "GET", ORIG_CHECK_URL, headers=headers, data={}
    )
    if response.status_code == 200:
        return True
    return False


def gpt_auth_check(key: str) -> bool:
    client = OpenAI(
        api_key=key
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Say this is a test",
                }
            ],
            model="gpt-3.5-turbo",
        )
    except openai.AuthenticationError:
        return False
    if not chat_completion:
        return False
    return True


def smart_truncate(text: str, max_chars=512, max_chunks=8):
    """
    Intelligently truncate a text below a maximum number of chars, and chunks.
    Splits on line ends to ensure, that all chunks end in a logical fashion.
    :param text: text to split.
    :param max_chars: max number of chars in each chunk.
    :param max_chunks: maximum number of chunks.
    :return: Text in chunks up to the max size.
    """
    sentences = text.split(".")
    current_chars = 0
    truncated_paragraph = ""
    chunks = []
    chunk = ""

    if len(text) < max_chars:
        return [text]

    for sentence in sentences:
        if len(sentences) > max_chars:
            return chunks
        if current_chars + len(sentence) <= max_chars:
            chunk += sentence
            current_chars += len(sentence)
        else:
            chunks.append(chunk)
            if len(chunks) == max_chunks:
                return chunks
            chunk = sentence
            current_chars = len(chunk)

    return chunks


def shannon_entropy(string: str):
    """
    Calculate the shannon entropy of a given string.
    :param string: string to measure.
    :return: entropy of string.
    """
    # Calculate the frequency of each character in the string
    frequencies = Counter(string)

    # Calculate the probability of each character
    probabilities = [float(freq) / len(string) for freq in frequencies.values()]

    # Calculate the Shannon entropy
    entropy = -sum(p * math.log2(p) for p in probabilities if p != 0)

    return entropy


def split_mappable(text, chunk_max, max_chunks, real_label) -> dict:
    """
    Mappable version of split function, takes same args.
    :param text:
    :param chunk_max:
    :param max_chunks:
    :param real_label:
    :return:
    """
    return {
        "human_chunks": smart_truncate(text[real_label], chunk_max, max_chunks)}

