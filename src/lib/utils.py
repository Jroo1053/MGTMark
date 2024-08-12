"""

MGTMark - Machine Generated Text Detection & Obfuscation Benchmarking Tool.

Copyright (C) 2024 Elyse Frary

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
Utility Functions
"""

import pyximport

pyximport.install()
from src.lib.models import RunConfig, AttackMethod, MGTDataset, \
    ClassifierPipeline
import json
from src.lib.mappables import (misspell_mappable, glyph_attack_mappable,
                               zwsp_padding_mappable, paragraph_mappable,
                               strat_space_mappable, alter_numbers_mappable,
                               whitespace_mappable)

SUPPORTED_METHODS = {
    "spelling": misspell_mappable,
    "glyph": glyph_attack_mappable,
    "zwsp": zwsp_padding_mappable,
    "spacing": strat_space_mappable,
    "alter_number": alter_numbers_mappable,
    "whitespace": whitespace_mappable,
    "paragraph": paragraph_mappable,
    "article_delete": "",
    "translate": "",
    "paraphrase": "",
}

CHANCE_ONLY_ATTACKS = [
    "spacing", "alter_number", "whitespace", "paragraph",
]
SUPPORTED_APIS = [
    "ORIG"
]


def _config_sanity_checks(config_path: str) -> bool:
    """
    Run basic sanity checks against config file, before running config setup.
    :param config_path: candidate configuration file.
    :return: true if checks pass.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            config_json = json.loads(config_file.read())
    except IOError as exc:
        print(f"Got file error, while loading config: {config_path}. {exc}")
        return False
    except json.JSONDecodeError as js_exc:
        print(
            f"Got json decode error, while loading config: {config_path}. {js_exc}")
        return False

    if "datasets" not in config_json.keys() or "models" not in config_json.keys():
        print(f"Config {config_path} is missing datasets or classifiers")
        return False
    config_method_names = set([x["name"] for x in config_json["attacks"]])
    if set_diff := config_method_names.difference(
            set(SUPPORTED_METHODS.keys())):
        print(f"Got unsupported method(s): {set_diff}, exiting!")
        return False
    return True


def load_config(config_path: str, max_samples: int) -> RunConfig | bool:
    """
    Load configuration from file and produce RunConfig to support later actions.
    :param config_path: path to config file.
    :param max_samples: maximum number of samples to parse from each dataset.
    :return: RunConfig of options.
    """
    if not _config_sanity_checks(config_path):
        return False

    with open(config_path, "r", encoding="utf-8") as conf_file:
        config_json = json.loads(conf_file.read())

    attack_methods = []
    new_method = None
    for method in config_json["attacks"]:
        method_name = method["name"].lower()
        if "chance" in method.keys():
            method_chance = method["chance"]
        if method_name in CHANCE_ONLY_ATTACKS:
            new_method = AttackMethod(
                name=method_name,
                attack_function=SUPPORTED_METHODS[method_name],
                attack_args={
                    "chance": method_chance
                }
            )
        elif method_name == "glyph":
            with open(method["pair_file"], "r") as pair_file:
                pairs_json = json.loads(pair_file.read())
                pair_map = {x["base"]: x["alts"] for x in pairs_json}
            new_method = AttackMethod(
                name=method_name,
                attack_function=SUPPORTED_METHODS[method_name],
                attack_args={
                    "pairs": pair_map,
                    "chance": method_chance
                }
            )
        elif method_name == "spelling":
            with open(method["spell_file"], "r") as spell_file:
                pairs_json = json.loads(spell_file.read())
                spell_map = {x["base"]: x["alts"] for x in pairs_json}
            new_method = AttackMethod(
                name="spelling",
                attack_function=SUPPORTED_METHODS["spelling"],
                attack_args={
                    "chance": method["chance"],
                    "min_len": method["min_len"],
                    "pairs": spell_map
                }
            )
        elif method_name == "zwsp":
            new_method = AttackMethod(
                name=method_name,
                attack_function=SUPPORTED_METHODS[method_name],
                attack_args={
                    "padding_mult": method["padding_mult"]
                }
            )
        if new_method:
            attack_methods.append(new_method)
    datasets = []
    classifiers = []

    for dataset in config_json["datasets"]:
        datasets.append(
            MGTDataset(
                path=dataset["name"],
                mgt_label=dataset["machine_samples"],
                human_label=dataset["human_samples"],
                max_samples=max_samples
            )
        )
    for classifier in config_json["models"]:
        is_api = False
        if classifier["name"] in SUPPORTED_APIS:
            is_api = True
        classifiers.append(
            ClassifierPipeline(
                name=classifier["name"],
                mgt_label=classifier["machine_label"],
                human_label=classifier["human_label"],
                is_api=is_api
            )
        )

    attack_runs = []
    for attack_run in config_json["attack_configs"]:
        if attack_run == ["*"]:
            for x in attack_methods:
                attack_runs.append([x])
        else:
            new_run = []
            for attack in attack_run:
                if attack in SUPPORTED_METHODS.keys():
                    matching_method = [x for x in attack_methods if
                                       x.name == attack.lower()]
                    if matching_method:
                        new_run.append(
                            attack_methods[
                                attack_methods.index(matching_method[0])]
                        )
            attack_runs.append(new_run)

    return RunConfig(
        datasets=datasets,
        classifiers=classifiers,
        attacks=attack_methods,
        max_samples=max_samples,
        runs=attack_runs
    )
