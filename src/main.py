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
import json
import logging
import os
import random
import secrets
import statistics
import time
import warnings

import textstat
from line_profiler_pycharm import profile
from prettytable import PrettyTable
from tqdm import tqdm

from src.lib.attacks import SUPPORTED_METHODS, load_methods, prep_dataset
from src.lib.models import APIPipeline
from src.lib.tf_utils import cuda_check
from src.lib.utils import shannon_entropy

"""
Disable annoying warnings from tensorflow + huggingface.Cant fix these issues
without re-training all the models so this is the best option.
"""
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)

from argparse import ArgumentParser
from datetime import datetime
from transformers import pipeline
from datasets import load_from_disk, load_dataset

if not cuda_check():
    print("CUDA Not Available, Exiting!")
    exit(1)

"""
BEGIN GLOBALS
"""

SPLITTER = "-" * 60
TITLE = "LLM Test - LLM Evasion And Detection Test System"
CHUNK_KEYS = ["human", "generated", "glyph", "zwsp", "strat", "spelling"]
DATA_OUTPUT_PATH = "data/results/"
DATA_PATH = "data/done"
MACHINE_TAG = "llm_base"
HUMAN_TAG = "human_base"

SUPPORTED_DETECT_APIS = [
    "ORIG", "WINSTON"
]

"""
END GLOBALS
"""


def main():
    """
    LLMTest - LLM Evasion and Detection Test System.
    """
    print(f"{SPLITTER}\n{TITLE}\n{SPLITTER}")
    if args.config_file:
        results = run_with_config(args.config_file)
        analyse_results(results)
    else:
        if args.key_file:
            results = test_with_dataset(
                dataset=args.base_data,
                model=args.detect_model,
                max_samples=args.samples,
                auth_creds=args.key_file
            )
        else:
            results = test_with_dataset(
                dataset=args.base_data,
                model=args.detect_model,
                max_samples=args.samples
            )
        analyse_results(results)




def run_with_config(config_path: str) -> dict:
    """
    Run against multiple files rather than one, using the specified config.
    :rtype: dict raw results for later analysis
    :param config_path: config file to load from.
    """
    with open(config_path, "r", encoding="utf-8") as config_file:
        pairs_json = json.loads(config_file.read())

    methods = load_methods(
        pairs_json["attacks"]
    )

    prepped_data = []

    for dataset in pairs_json["datasets"]:
        base_data = load_dataset(
            dataset["name"], split="train"
        ).shuffle(
            seed=random.randint(0, 10 ** 5)
        )
        if not args.samples:
            base_data = base_data.select(
                [i for i in range(pairs_json["samples"])]
            )
        else:
            base_data = base_data.select(
                [i for i in range(args.samples)]
            )
        """
        Setup every dataset by applying attacks and splitting into a consistent 
        format. Write to disk so that results can be replicated if needed. 
        Attacks and sampling are random so reusing same options/dataset will not
        work fully. 
        """
        # Append random string to results so they can be better identified.
        rand_end = secrets.token_urlsafe(8)
        base_data = prep_dataset(
            base_data=base_data,
            method_config=methods,
            real_label=dataset["human_samples"],
            machine_label=dataset["machine_samples"]
        )


        set_name = dataset["name"].split("/")[1]
        file_path = os.path.join(
            DATA_PATH, set_name
        )
        final_path = f"{file_path}/{set_name}_{rand_end}.jsonl"
        base_data.to_json(final_path)
        prepped_data.append(
            final_path
        )

    all_results = []

    for dataset in prepped_data:
        for model in pairs_json["models"]:
            if "auth_file" not in model.keys():
                results = test_with_dataset(
                    dataset=dataset,
                    model=model["name"],
                    max_samples=pairs_json["samples"]
                )
            else:
                results = test_with_dataset(
                    dataset=dataset,
                    model=model["name"],
                    max_samples=pairs_json["samples"],
                    auth_creds=model["auth_file"]
                )
            """
            Add metadata to raw_results so that the analysis can be re-run. 
            Save result to disk for crash-recovery and replication. 
            """
            results["dataset"] = dataset
            results["model"] = model["name"]
            results["human_label"] = model["human_label"]
            results["machine_label"] = model["machine_label"]
            all_results.append(results)

            if "/" in model["name"]:
                model_name = model["name"].split("/")[1]
            else:
                model_name = model["name"]
            set_name = os.path.split(dataset)[-1]
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            filename = \
                f"raw_{model_name}_{set_name}_{len(results[HUMAN_TAG])}_{timestamp}_{rand_end}.csv"
            file_path = os.path.join(DATA_OUTPUT_PATH, filename)

            with open(file_path, "w") as tmp_out:
                tmp_out.write(str(results))
    return all_results


def analyse_results(all_results: list):
    """
    Print pretty tables given a list of results
    :param all_results: all results to analyse
    """
    """
    Get all datasets and attack methods in the result set, ditch any results
    with missing tags. 
    """
    all_results = [x for x in all_results if
                   "dataset" in x.keys() and "human_label" in x.keys()]
    all_results = sorted(all_results,key=lambda x:x["model"])
    all_datasets = set([x["dataset"] for x in all_results])
    all_methods = list(set([y for x in all_results for y in x.keys() if y in list(SUPPORTED_METHODS.keys()) + [HUMAN_TAG,MACHINE_TAG] ]))
    all_methods.sort()
    method_fprs = {}

    all_confs = []
    for method in all_methods:
        method_fprs[method] = []

    print(all_datasets)
    all_results_count = 0
    for dataset in all_datasets:
        dataset_results = [x for x in all_results if x["dataset"] == dataset]
        dataset_table = PrettyTable()
        table_header = ["Model/Attack"] + all_methods
        dataset_table.field_names = table_header

        for result_set in dataset_results:
            new_row = [
                result_set["model"]
            ]

            result_methods = [x for x in result_set if x in all_methods]
            result_methods.sort()
            for method in result_methods:
                try:
                    method_results = result_set[method]
                    method_results = [x for x in method_results if x]


                    method_human = [x for x in method_results if
                                    x["label"].lower() \
                                    == result_set["human_label"].lower() and x["score"] > .5]
                    method_machine = [x for x in method_results if
                                      x["label"].lower() \
                                      == result_set["machine_label"].lower() and x["score"] > .5]
                    all_confs += [x["score"] for x in method_results]

                    method_human = len(method_human)
                    method_machine = len(method_machine)
                    total_marked = method_human + method_machine
                    all_results_count += total_marked
                    if method == HUMAN_TAG:
                        try:
                            fpr = method_machine / total_marked
                        except ZeroDivisionError:
                            if method_human > method_machine:
                                fpr = 0
                            else:
                                fpr = 1
                    else:
                        try:
                            fpr = method_human / total_marked
                        except ZeroDivisionError:
                            if method_human > method_machine:
                                fpr = 1
                            else:
                                fpr = 0
                    method_fprs[method].append(fpr)
                    new_row.append(
                        round(fpr, 4)
                    )
                except Exception as exc:
                    print(f"failed to proccess result set: {result_set['dataset']}")
                    print(f"Got error: {exc}")
                try:
                    dataset_table.add_row(new_row)
                except ValueError:
                    continue

        print(SPLITTER)
        print(f"Results for {dataset}")
        print(dataset_table)
    print(SPLITTER)
    for method in all_methods:
        print(f"FPR for {method}: {statistics.mean(method_fprs[method])}")
        print(SPLITTER)
    print(SPLITTER)
    print(f"Mean confidence score: {statistics.mean(all_confs)}")
    print(f"Min confidence score: {min(all_confs)}")
    print(f"Plus 90 Count {len([x for x in all_confs if x > 0.95]) / len(all_confs) * 100 }")
    print(f"Confidence Score STDEV: {statistics.stdev(all_confs)}")



@profile
def test_with_dataset(dataset: str, model: str, max_samples: int,
                      auth_creds="") -> dict:
    """
    Run the dataset of AIGC samples against the LLM detector.
    :param dataset: Dataset of obfuscated AIGC samples.
    :param model: AGIC detection model to test with.
    :param max_samples: maximum number of samples.
    :return: dictionary of results.
    """
    """
    Don't load with streaming mode so that all chunks can be run through
    at the same time. Classifier runs slightly faster that way. 
    """
    try:
        base_data = load_from_disk(
            dataset
        ).shuffle(
            seed=random.randint(0, 10 ** 5)
        ).select(
            [i for i in range(max_samples)]
        )
    except IndexError:
        """
        if max_samples > len(dataset), load whole set.
        easier than actually geting len(dataset)
        """
        base_data = load_from_disk(
            dataset
        ).shuffle(
            seed=random.randint(0, 10 ** 5)
        )
    except FileNotFoundError:
        base_data = load_dataset(
            "json",
            data_files=dataset
        ).shuffle(
            seed=random.randint(0,10**5)
        )

    if model in SUPPORTED_DETECT_APIS:
        classifier_pipeline = APIPipeline(
            provider=model, auth_creds=auth_creds
        )
    else:
        classifier_pipeline = pipeline(task="text-classification", model=model,
                                       device=0)
    start_time = time.process_time()
    results_dict = {}

    # Get results for evasion methods
    all_methods = list(SUPPORTED_METHODS.keys()) + [HUMAN_TAG, MACHINE_TAG]

    print(SPLITTER)
    print(f"Results For: {dataset}")
    print(f"Detection Model: {model}")
    print(SPLITTER)

    for method in all_methods:
        chars = f"{method}_chars"
        if method in SUPPORTED_METHODS.keys():
            chunks = f"{method}_chunks"
        else:
            chunks = method
        if chunks in base_data.column_names['train']:
            if method not in [HUMAN_TAG,MACHINE_TAG]:
                method_chunks = [x[chunks] for x in base_data["train"] if x[chars] > 0]
            else:
                method_chunks = [x[chunks] for x in base_data["train"]]
            results_dict[method] = []
            print("Getting Results For: ", method)
            for chunk_data in tqdm([x for x in method_chunks if x]):
                chunk_res = classifier_pipeline(
                    chunk_data, max_length=512
                )
                if isinstance(classifier_pipeline, APIPipeline):
                    time.sleep(0.15)
                results_dict[method] += chunk_res
                results_dict[method + "_entropy"] = statistics.mean(
                    [shannon_entropy(x) for x in chunk_data])
                results_dict[method + "_readability"] = statistics.mean(
                    [textstat.flesch_reading_ease(x) for x in chunk_data])
            print(SPLITTER)

    total_time = time.process_time() - start_time
    print(f"Done in: {total_time} seconds")

    # Estimate time for different numbers of samples
    time_per_sample = total_time / max_samples

    time_magnitudes = [10 ** x for x in range(2, 10, 2)]
    for x in time_magnitudes:
        time_est = x * time_per_sample

        if time_est < 60:
            print(f"Would take {time_est:.2f} seconds for {x:,} samples")
        elif time_est < 3600:
            print(f"Would take {time_est / 60:.2f} minutes for {x:,} samples")
        else:
            print(f"Would take {time_est / 3600:.2f} hours for {x:,} samples")

    return results_dict


if __name__ == "__main__":
    """
    BEGIN ARGS
    """

    parser = ArgumentParser(description="AI Attack Tester")
    parser.add_argument("-a", "--ai-dataset",
                        help="Specify the AI dataset",
                        dest="base_data")
    parser.add_argument("-m", "--detection-model",
                        help="Detection Model to use",
                        dest="detect_model")
    parser.add_argument("-v", "--verbose",
                        action="store_true", help="Enable verbose mode")
    parser.add_argument("-s", "--samples",
                        type=int, default=10)
    parser.add_argument("-rl", "--real-label",
                        help="Specify the real label", default="Real")
    parser.add_argument("-ml", "--machine-label",
                        help="Specify the machine label", default="Fake")
    parser.add_argument("-c", "--config", help="config file",
                        dest="config_file")
    parser.add_argument("-k", "--key-file",
                        help="API key file, only applies when confi file is not used",
                        dest="key_file")
    args = parser.parse_args()

    """
    END ARGS
    """

    main()
