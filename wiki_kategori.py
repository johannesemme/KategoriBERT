# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WikiKategori dataset"""

import json
import os
import logger
import csv
import datasets

# You can copy an official description
_DESCRIPTION = """\
This dataset is designed to solve the task of categorizing a text wrt. 14 different categories obtained using the Wikipaedia category hierarchy.
"""
_CLASS_NAMES = [
    "Uddannelse",
    "Samfund",
    "Videnskab",
    "Natur",
    "Teknologi",
    "Kultur",
    "Historie"
    "Sundhed",
    "Geografi",
    "Økonomi",
    "Sport",
    "Religion",
    "Politik",
    "Erhvervsliv"
    ]


_BASE_DOWNLOAD_URL = "https://github.com/johannesemme/datasets/raw/main/wiki_kategori/"

class NewDataset(datasets.GeneratorBasedBuilder):
    """WikiKategori dataset"""

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "title": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "labels":datasets.Sequence(datasets.ClassLabel(names=_CLASS_NAMES)),
                }
            )
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(os.path.join(_BASE_DOWNLOAD_URL, "train.csv"))
        val_path = dl_manager.download_and_extract(os.path.join(_BASE_DOWNLOAD_URL, "val.csv"))
        test_path = dl_manager.download_and_extract(os.path.join(_BASE_DOWNLOAD_URL, "test.csv"))
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": [train_path]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": [val_path]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": [test_path]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            csv_reader = csv.DictReader(f, delimiter=",", fieldnames=list(self.config.features.keys()))
            for row_idx, row in enumerate(csv_reader):
                row["labels"] = [int(ind) for ind in row["labels"].split(",")]
                yield row_idx, row
