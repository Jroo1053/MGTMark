"""

MGTMark - Machine Generated Text Detection & Obfuscation Benchmarking Tool.

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


import os
from dataclasses import dataclass
import random
from typing import Callable

from datasets import load_dataset

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

    def __repr__(self):
        return self.name

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
        self._fix_indexes()

    def _fix_indexes(self):
        pass

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

    def __repr__(self):
        return str(self.attack_type)
