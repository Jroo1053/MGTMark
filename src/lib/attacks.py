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
import time

"""
Attack methods and associated wrapper function.
"""

import random

from openai import OpenAI

from line_profiler_pycharm import profile
from transformers import pipeline
from datasets import Dataset

from src.lib.utils import smart_truncate, smart_truncate, load_auth_creds, \
    split_mappable
import json
import os

TOKEN_MAX = 512
MAX_CHUNKS = 8
SPLITTER = "-" * 80

def translate(text: str, base_lang: str, new_lang: str,model: str, auth_key: str) -> tuple[
    str,int]:
    client = OpenAI(
        api_key=auth_key
    )
    base_chat = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Translate the following into {new_lang}: {text}"
            }
        ],
        model=model
    ).choices[0].message.content
    time.sleep(0.25)
    base_chat = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Translate the following into {base_lang}: {base_chat}"
            }
        ],
        model=model
    ).choices[0].message.content

    return base_chat,len(text) - len(base_chat)

def translate_mappable(entry:str,args:dict,machine_label:slice) -> dict:
    rand_timer = random.randint(1, 5)
    time.sleep(float(rand_timer) / 10)
    base_result, chars_swapped = translate(
        text=entry[machine_label],base_lang=args["base_lang"],
        new_lang=args["new_lang"],model=args["model"],auth_key=args["key"]
    )
    map_result = {
        "translate_chunks": smart_truncate(base_result, TOKEN_MAX, MAX_CHUNKS),
        "translate_chars": chars_swapped
    }
    return map_result


def paraphrase(text: str, prompt: str, model: str, auth_key: str) -> tuple[
    str, int]:
    client = OpenAI(
        api_key=auth_key
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{prompt} {text}",
            }
        ],
        model=model,
    ).choices[0].message.content
    return chat_completion, len(text) - len(chat_completion)


def paraphrase_mappable(entry: str, args: dict, machine_label: slice) -> dict:
    rand_timer = random.randint(1, 5)
    time.sleep(float(rand_timer) / 10)
    base_result, chars_swapped = paraphrase(
        entry[machine_label], prompt=args["prompt"], model=args["model"],
        auth_key=args["key"]
    )
    map_result = {
        "paraphrase_chunks": smart_truncate(base_result, TOKEN_MAX, MAX_CHUNKS),
        "paraphrase_chars": chars_swapped
    }
    return map_result


def misspell(text: str, pairs: dict, chance: float) -> tuple[str, int]:
    """
    Add spelling mistakes to a text, given a set of base words and misspelled
    alternatives.
    :param text: base text to modify.
    :param pairs: set of base words and alternatives.
    :return: new text with spelling errors.
    """
    new_string = []
    chars_swapped = 0
    for word in text.split():
        alternatives = pairs.get(word)

        if alternatives and random.random() <= chance and len(word) > 4:
            swap = random.choice(alternatives)
            new_string.append(swap)
            chars_swapped += len(swap)
        else:
            new_string.append(word)

    return ' '.join(new_string), chars_swapped


def misspell_mappable(entry: str, args: dict, machine_label: slice) -> dict:
    """
    Wrapper func for the misspelling attack, so that it can be applied via
    HuggingFace map().
    :param entry: entry to test
    :param args: dict of args
    :param machine_label: label used to mark machine generated contents from
    label.
    :return: dict with new text, and number of changed chars.
    """
    base_result, chars_swapped = misspell(
        entry[machine_label], args["pairs"], args["chance"]
    )
    map_result = {
        "spelling_chunks": smart_truncate(base_result, TOKEN_MAX, MAX_CHUNKS),
        "spelling_chars": chars_swapped
    }
    return map_result


def strat_space(text: str, chance: float) -> tuple[str, int]:
    """
    Obfuscate text by strategically inserting spaces after commas.
    :param text: text to change
    :param chance: chance of attack occurring on every index of ','
    :return: new text, chars added.
    """
    if "," not in text:
        return text, 0
    new_text = [*text]
    commas = [i for i, x in enumerate(new_text) if x == ","]
    chars_swapped = 0
    for char_index in range(len(commas)):
        if random.random() <= chance:
            new_indexes = [i for i, x in enumerate(new_text) if x == ","]
            new_text.insert(
                new_indexes[chars_swapped], " "
            )
            chars_swapped += 1
    return "".join(new_text), chars_swapped


def strat_space_mappable(entry, args, machine_label) -> dict:
    """
    Wrapper func for the spacing attack, so that it can be applied via
    HuggingFace map().
    :param entry: entry to test
    :param args: dict of args
    :param machine_label: label used to mark machine generated contents from
    label.
    :return: dict with new text, and number of changed chars.
    """
    base_result, chars_swapped = strat_space(
        entry[machine_label], args["chance"]
    )
    map_result = {
        "spacing_chunks": smart_truncate(base_result, TOKEN_MAX, MAX_CHUNKS),
        "spacing_chars": chars_swapped
    }
    return map_result


def zwsp_padding(text: str, padding_multiplier=.1,
                 zwsp_chars=["\u200B", "\u200C", "\u200d"]) -> tuple[str, int]:
    """
    Insert a number of Zero Width Spaces into the given string.
    Insert Len(string) * padding_multiplier characters.
    :param text: text to modify.
    :param padding_multiplier: number of ZWSP chars to add
    relative to the size of the base text.
    :param zwsp_chars: List of ZWSP chars to use.
    :return: new text number of chars changed
    """
    new_text = [*text]
    zwsp_percent = int(len(new_text) * padding_multiplier)
    for x in range(zwsp_percent):
        new_text.insert(
            random.randint(0, len(new_text)), random.choice(zwsp_chars)
        )
    return "".join(new_text), zwsp_percent


def zwsp_padding_mappable(entry, args, machine_label) -> dict:
    """
    Wrapper func for the ZWSP padding attack, so that it can be applied via
    HuggingFace map().
    :param entry: entry to test
    :param args: dict of args
    :param machine_label: label used to mark machine generated contents from
    label.
    :return: dict with new text, and number of changed chars.
    """
    base_result, chars_swapped = zwsp_padding(
        entry[machine_label], args["padding_mult"]
    )
    map_result = {
        "zwsp_chunks": smart_truncate(base_result, TOKEN_MAX, MAX_CHUNKS),
        "zwsp_chars": chars_swapped
    }
    return map_result


def glyph_attack(text: str, pairs: dict, glyph_chance: float) -> tuple[
    str, int]:
    """
    Run homoglyph against a text, given a list of homoglyphs.
    :param text: text to modify.
    :param pairs: list of glyphs and pairs.
    :return: modified text.
    """
    new_string = []
    chars_swapped = 0
    for char in text:
        alternatives = pairs.get(char)

        if alternatives and random.random() <= glyph_chance:
            swap = random.choice(alternatives)
            new_string.append(swap)
            chars_swapped += len(swap)
        else:
            new_string.append(char)

    return ''.join(new_string), chars_swapped


def glyph_attack_mappable(entry, args: dict, machine_label: str) -> dict:
    """
    Wrapper func for the homoglyph, so that it can be applied via
    HuggingFace map().
    :param entry: entry to test
    :param args: dict of args
    :param machine_label: label used to mark machine generated contents from
    label.
    :return: dict with new text, and number of changed chars.
    """
    base_result, chars_swapped = glyph_attack(
        entry[machine_label], args["pairs"], args["chance"]
    )
    map_result = {
        "glyph_chunks": smart_truncate(base_result, TOKEN_MAX, MAX_CHUNKS),
        "glyph_chars": chars_swapped
    }
    return map_result


"""
Define GLOBAL of all attacks, need by other files but also needs to be defined
after attacks, so that's why its in this odd spot.
"""

SUPPORTED_METHODS = {
    "spelling": misspell_mappable,
    "glyph": glyph_attack_mappable,
    "zwsp": zwsp_padding_mappable,
    "spacing": strat_space_mappable,
    "paraphrase": paraphrase_mappable,
    "translate": translate_mappable
}


def load_methods(method_file: str) -> dict:
    """
    Load attack methods from given config file, return processed dict
    also loads attack datasets.
    :param method_file: file to load methods from
    :return: dict of methods, setup for running attacks.
    """
    if isinstance(method_file, str):
        with open(method_file, "r") as method_json:
            method_dict = json.loads(method_json.read())
    else:
        method_dict = method_file
    file_methods = set([x["name"] for x in method_dict])
    if set_dif := file_methods.difference(set(SUPPORTED_METHODS.keys())):
        print(f"Got unsupported method(s): {set_dif}, exiting!")

    for method in method_dict:
        if method["name"] == "glyph":
            # Load glyphs from config file.
            with open(method["pair_file"], "r") as pair_file:
                pairs_json = json.loads(pair_file.read())
            pair_map = {x["base"]: x["alts"] for x in pairs_json}
            method["pairs"] = pair_map
        if method["name"] == "spelling":
            pairs = []
            # Load spelling dict
            with open(method["spell_file"], "r", ) as spell_file:
                current_base = ""
                tmp_pairs = []
                for current_line in spell_file:
                    # Base words are marked with $
                    if current_line.startswith("$"):
                        pairs.append(
                            {
                                "base": current_base,
                                "pairs": tmp_pairs
                            }
                        )
                        current_base = current_line[1:-1].lower()
                        tmp_pairs = []
                    else:
                        tmp_pairs.append(current_line[:-1].lower())
                # Convert to map for performance boost
                method["pairs"] = {x["base"]: x["pairs"] for x in pairs}
        if method["name"] in ["paraphrase","translate"]:
            method["key"] = load_auth_creds(method["auth_file"], model="GPT3")

    return method_dict


@profile
def prep_dataset(base_data, method_config: dict,
                 real_label: str, machine_label: str) -> Dataset:
    """
    Setup base data by running each attack against it
    :param base_data: dataset to modify
    :param method_config: dict of attack methods and options
    :param real_label: label to mark human text
    :param machine_label: label to mark MGT
    :return: new dataset
    """
    proc_count = os.cpu_count() - 2
    for method in method_config:
        # Get method function from SUPPORTED_METHODS
        method_func = SUPPORTED_METHODS[method["name"]]
        if method_func:
            print("Running Attack: ", method["name"])
            not_mappable = False
            if method_func in [paraphrase_mappable,translate_mappable]:
                not_mappable = True
            if not_mappable:
                base_data = base_data.map(
                    lambda x: method_func(x, method, machine_label),
                )
            elif not not_mappable and len(base_data) >= 1000:
                base_data = base_data.map(
                    lambda x: method_func(x, method, machine_label),
                    num_proc=proc_count
                )
            else:
                base_data = base_data.map(
                    lambda x: method_func(x, method, machine_label)
                )
            print(SPLITTER)

    # Always load from human and llm_base tags
    for method in [machine_label, real_label]:
        if len(base_data) >= 1000:
            base_data = base_data.map(
                lambda x: split_mappable(
                    x, 256, 16, method),
                num_proc=proc_count
            )
        else:
            base_data = base_data.map(
                lambda x: split_mappable(
                    x, 256, 16, method)
            )
    print(SPLITTER)

    for method in method_config:
        method_tag = method["name"] + "_chunks"
        method_chars = method["name"] + "_chars"
        null_results = [x for x in base_data[method_chars] if x == 0]
        print(f"Got: {len(null_results)} null attacks for attack {method_tag}")

    # Change columns for later use.
    base_data = base_data.rename_column(
        real_label, "human_base"
    )
    base_data = base_data.rename_column(
        machine_label, "llm_base"
    )

    return base_data
