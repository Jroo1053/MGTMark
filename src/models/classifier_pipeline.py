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


from tqdm import tqdm
from transformers import pipeline

from src.models.dataset import MGTDataset


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

    def __repr__(self):
        return self.name

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

    def __repr__(self):
        return str({
            self.classifier.name: self.is_flagged_mgt
        })

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
