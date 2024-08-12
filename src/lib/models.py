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

# Models
import os
import random
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline

PROC_COUNT = os.cpu_count() - 2


@dataclass
class AttackMethod:
    """
    Representation of every authorship obfuscation attack.
    :param name: name of the attack, must match one in config file.
    :param attack_function: function that applies attack.
    :param attack_args: dict of args to apply to attack function.
    :param is_mappable: toggles multithreading off/on.
    """
    name: str
    attack_function: Callable
    attack_args: dict
    is_mappable: bool = True


class MGTDataset:
    """
    Representation of a dataset containing MGT and human samples.
    """

    def __init__(self, path: str, mgt_label: str, human_label: str,
                 max_samples: int):
        """
        :param path: path of dataset as set in config.
        :param mgt_label: column used to store MGT.
        :param human_label: column used to store human texts.
        """
        self.name = path
        self.mgt_label = mgt_label
        self.human_label = human_label
        self.text_samples = []
        self.data = {}
        self.max_samples = max_samples

    def load_data(self, attacks: list[list[AttackMethod]]):
        """
        Load data from dataset path, apply attacks and convert to TextSamples
        :param attacks: attacks to apply.
        """
        base_data = load_dataset(
            path=self.name, split="train"
        ).shuffle(
            seed=random.randint(0, 10 ** 5)
        )
        if self.max_samples > 0:
            base_data = base_data.select(
                list(range(self.max_samples))
            )

        for label in [self.mgt_label, self.human_label]:
            if label == self.mgt_label:
                new_label = "mgt_chunks"
            else:
                new_label = "human_chunks"
            for sample in base_data[label]:
                self.text_samples.append(
                    TextSample(
                        attack_type=new_label,
                        content=sample
                    )
                )
        for attack_run in attacks:
            attack_index = 0
            for attack in attack_run:
                if attack_index == 0:
                    data_column = self.mgt_label
                else:
                    data_column = attack_run[attack_index - 1].name + "_chunks"

                if callable(attack.attack_function):
                    print("-" * 40 + f"\n Running Attack: {attack.name}")
                    if attack.is_mappable:
                        base_data = base_data.map(
                            lambda x: attack.attack_function(
                                x, attack.attack_args,
                                data_column),
                            num_proc=PROC_COUNT
                        )
                    else:
                        base_data = base_data.map(
                            lambda x: attack.attack_function(
                                x, attack.attack_args,
                                data_column),
                        )
                attack_index += 1
            data_column = attack_run[-1].name + "_chunks"
            for attack_sample in base_data[data_column]:
                self.text_samples.append(
                    TextSample(
                        attack_type=[x.name for x in attack_run],
                        content=attack_sample
                    )
                )


@dataclass
class TextSample:
    """
    Representation of a text document.
    :param attack_type: attack used to obfuscate text,
     can also be base human or MGT.
    :param content: truncated samples.
    """
    attack_type: list | str
    content: str

    def __post_init__(self):
        if not isinstance(self.attack_type, list):
            self.attack_type = [self.attack_type]
        self.classification_results = []
        self.classifier = None
        self.entropy = 0
        self.attack_indexes = []


class ClassifierPipeline:
    """
    Representation of a classification system used to determine MGT authorship.
    """

    def __init__(self, name: str, mgt_label: str, human_label: str,
                 is_api: bool):
        """
        :param name: name of classifier must match config.
        :param mgt_label: label used by classifier to mark mgt entries.
        :param human_label: label used by classifier to mark human entries.
        :param is_api: is HTTP API.
        """
        self.name = name
        self.mgt_label = mgt_label
        self.human_label = human_label
        self.is_api = is_api
        if self.is_api:
            pass
        else:
            self.pipeline = pipeline(
                task="text-classification", model=name,
                device=0
            )

    def apply_classifier(self, dataset: MGTDataset):
        """
        Apply classifier to all entries in the dataset and append results
        to TextSamples
        :param dataset: dataset to test.
        """
        for sample_text in tqdm(dataset.text_samples):
            sample_res = self.pipeline(
                sample_text.content, max_length=512
            )
            sample_text.classification_results.append(
                ClassificationResult(
                    sample_res, self
                )
            )


class ClassificationResult:
    """
    Representation of the results of a classification test.
    """

    def __init__(self, raw_results: list[dict], classifier: ClassifierPipeline):
        """
        :param raw_results: results straight from the classifier.
        :param classifier: classifier to associate with.
        """
        self.results = raw_results
        self.classifier = classifier
        self.is_flagged_mgt = None
        self.resolve_results()

    def resolve_results(self):
        """
        If the raw results are more than one entry and are not conclusive,
        then resolve to use the most common option.
        """
        if len(self.results) > 1:
            self.is_single_result = False
            mgt_count = len([x for x in self.results if
                             x["label"] == self.classifier.mgt_label])
            human_count = len([x for x in self.results if
                               x["label"] == self.classifier.human_label])
            if mgt_count >= human_count:
                self.is_flagged_mgt = True
            else:
                self.is_flagged_mgt = False
        elif self.results:
            self.is_single_result = True
            if self.results[0]["label"] == self.classifier.mgt_label:
                self.is_flagged_mgt = True
            else:
                self.is_flagged_mgt = False


class RunConfig:

    def __init__(self, classifiers: list[ClassifierPipeline],
                 attacks: list[AttackMethod], datasets: list[MGTDataset],
                 runs: list[list[AttackMethod]], max_samples=1000):
        self.classifiers = classifiers
        self.attacks = attacks
        self.datasets = datasets
        self.runs = runs
        self.max_samples = max_samples

    def load_data(self):
        for dataset in self.datasets:
            dataset.load_data(attacks=self.runs)

    def run_classifiers(self):
        for classifier in self.classifiers:
            for dataset in self.datasets:
                classifier.apply_classifier(dataset)

    def get_results(self):
        summary_data = []
        for dataset in self.datasets:
            for y in range(len(self.classifiers)):
                classifier = self.classifiers[y]
                for attack_run in self.runs:
                    attack = [x.name for x in attack_run]
                    mgt_total = len(
                        [x for x in dataset.text_samples if
                         x.classification_results[y].is_flagged_mgt
                         and x.attack_type == attack]
                    )
                    human_total = len(
                        [x for x in dataset.text_samples
                         if not x.classification_results[y].is_flagged_mgt
                         and x.attack_type == attack]
                    )
                    all_total = mgt_total + human_total

                    if all_total > 0:
                        if attack == "human_chunks":
                            fpr = mgt_total / all_total
                        else:
                            fpr = human_total / all_total
                    else:
                        fpr = 0

                    summary_data.append({
                        "Classifier": classifier.name,
                        "Dataset": dataset.name,
                        "Attack": attack,
                        "MGT": mgt_total,
                        "Human": human_total,
                        "Total": all_total,
                        "FPR": fpr
                    })
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
