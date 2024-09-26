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

import pandas as pd

from src.models.classifier_pipeline import ClassifierPipeline
from src.models.dataset import AttackMethod, MGTDataset


class RunConfig:
    """
    Object representing all config options, updated throughout the process.
    """

    def __init__(self, classifiers: list[ClassifierPipeline],
                 attacks: list[AttackMethod], datasets: list[MGTDataset],
                 runs: list[list[AttackMethod]], max_samples=1000):
        """
        :param classifiers: list of classifiers.
        :param attacks:  list of attacks, note attack configs not attack runs.
        :param datasets: list of datasets.
        :param runs: list of attack runs.
        :param max_samples: maximum number of samples to load.
        """
        self.classifiers = classifiers
        self.attacks = attacks
        self.datasets = datasets
        self.runs = runs
        self.max_samples = max_samples

    def load_data(self):
        """
        Load data for every dataset and convert into TextSamples
        """
        for dataset in self.datasets:
            dataset.load_data(attacks=self.runs)

    def run_classifiers(self):
        """
        Apply every classifier to every sample in every dataset.
        """
        for classifier in self.classifiers:
            for dataset in self.datasets:
                classifier.apply_classifier(dataset)

    def get_results(self):
        """
        Present results of classification attacks.
        """
        summary_data = []

        base_attacks = [
            [
                AttackMethod(name="mgt_chunks", attack_function=None,
                             attack_args=None)
            ],
            [
                AttackMethod(name="human_chunks", attack_function=None,
                             attack_args=None)
            ]
        ]

        for dataset in self.datasets:
            for y in range(len(self.classifiers)):
                classifier = self.classifiers[y]
                for attack_run in self.runs + base_attacks:
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
                        if attack == ["human_chunks"]:
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
