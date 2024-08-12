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

Utility Functions
"""
import json
import pyximport
import charset_normalizer as cn

pyximport.install()

from src.lib.models import (RunConfig, AttackMethod, MGTDataset,
                            ClassifierPipeline)

from src.lib.mappables import (misspell_mappable, glyph_attack_mappable,
                               zwsp_padding_mappable, paragraph_mappable,
                               strat_space_mappable, alter_numbers_mappable,
                               whitespace_mappable,article_mappable,upper_lower_mappable)

SUPPORTED_METHODS = {
    "spelling": misspell_mappable,
    "glyph": glyph_attack_mappable,
    "zwsp": zwsp_padding_mappable,
    "spacing": strat_space_mappable,
    "alter_number": alter_numbers_mappable,
    "whitespace": whitespace_mappable,
    "paragraph": paragraph_mappable,
    "article": article_mappable,
    "upper_lower":upper_lower_mappable,
    "translate": "",
    "paraphrase": "",
}

CHANCE_ONLY_ATTACKS = [
    "spacing", "alter_number", "whitespace", "paragraph","upper_lower"
]
SUPPORTED_APIS = [
    "ORIG"
]


def _guess_encoding(path: str) -> str:
    with open(path, "rb") as code_test:
        data = code_test.read(10 ** 6)
    return cn.detect(data).get("encoding")


def _get_json(config_path: str) -> bool | dict:
    """
    Run basic sanity checks against config file, before running config setup.
    :param config_path: candidate configuration file.
    :return: true if checks pass.
    """
    try:
        with open(config_path, "r",
                  encoding=_guess_encoding(config_path)) as config_file:
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
    return config_json


def _load_datasets(config_json: dict, max_samples: int) -> list[MGTDataset]:
    """

    :param config_json:
    :param max_samples:
    :return:
    """
    datasets = []
    for dataset in config_json["datasets"]:
        datasets.append(
            MGTDataset(
                path=dataset["name"],
                mgt_label=dataset["machine_samples"],
                human_label=dataset["human_samples"],
                max_samples=max_samples
            )
        )
    return datasets


def _load_classifiers(config_json: dict) -> list[ClassifierPipeline]:
    """

    :param config_json:
    :return:
    """
    classifiers = []
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
    return classifiers


def _load_attacks(config_json: dict) -> list[AttackMethod]:
    """

    :param config_json:
    :return:
    """
    attack_methods = []
    new_method = None

    # Load config methods, no pretty way to do this.
    for method in config_json["attacks"]:
        method_name = method["name"].lower()
        method_chance = method.get("chance")

        if method_name in CHANCE_ONLY_ATTACKS:
            new_method = AttackMethod(
                name=method_name,
                attack_function=SUPPORTED_METHODS[method_name],
                attack_args={
                    "chance": method_chance
                }
            )
        elif method_name == "glyph":
            with open(method["pair_file"], "r", encoding=_guess_encoding(
                    method["pair_file"])) as pair_file:
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
            with open(method["spell_file"], "r", encoding=_guess_encoding(
                    method["spell_file"])) as spell_file:
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
        elif method_name == "article":
            new_method = AttackMethod(
                name=method_name,
                attack_function=SUPPORTED_METHODS[method_name],
                attack_args={
                    "articles":method["articles"],
                    "chance":method_chance
                }
            )
        if new_method:
            attack_methods.append(new_method)
    return attack_methods


def _load_attack_runs(config_json: dict, attack_methods: list[AttackMethod]) -> \
list[[AttackMethod]]:
    """
    :param config_json:
    :param attack_methods:
    :return:
    """
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
    return attack_runs


def load_config(config_path: str, max_samples: int) -> RunConfig | bool:
    """
    Load configuration from file and produce RunConfig to support later actions.
    :param config_path: path to config file.
    :param max_samples: maximum number of samples to parse from each dataset.
    :return: RunConfig of options.
    """
    config_json = _get_json(config_path)
    if not config_json:
        return False

    datasets = _load_datasets(config_json, max_samples)
    classifiers = _load_classifiers(config_json)
    attack_methods = _load_attacks(config_json)
    attack_runs = _load_attack_runs(config_json, attack_methods)

    return RunConfig(
        datasets=datasets,
        classifiers=classifiers,
        attacks=attack_methods,
        max_samples=max_samples,
        runs=attack_runs
    )
