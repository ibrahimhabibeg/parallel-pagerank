import os
import requests
import gzip
import pandas as pd
from scipy.sparse import csr_array
import numpy as np

dataset_link = {
    "web-Google": "https://snap.stanford.edu/data/web-Google.txt.gz",
    "web-BerkStan": "https://snap.stanford.edu/data/web-BerkStan.txt.gz",
    "web-Stanford": "https://snap.stanford.edu/data/web-Stanford.txt.gz",
    "web-NotreDame": "https://snap.stanford.edu/data/web-NotreDame.txt.gz",
}

dataset_file_name = {
    "web-Google": "web-Google.txt",
    "web-BerkStan": "web-BerkStan.txt",
    "web-Stanford": "web-Stanford.txt",
    "web-NotreDame": "web-NotreDame.txt",
}

dataset_number_unique_nodes = {
    "web-Google": 875_713,
    "web-BerkStan": 685_230,
    "web-Stanford": 281_903,
    "web-NotreDame": 325_729,
}

dataset_number_edges = {
    "web-Google": 5_105_039,
    "web-BerkStan": 7_600_595,
    "web-Stanford": 2_312_497,
    "web-NotreDame": 1_497_134,
}

supported_datasets = list(dataset_link.keys())

DATA_FOLDER = "data"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)


class SnapDataManager:
    def __init__(self, dataset_name):
        assert dataset_name in dataset_link, f"Dataset '{dataset_name}' not supported."
        self.dataset_name = dataset_name
        self.dataset_url = dataset_link[dataset_name]
        self.dataset_file_path = os.path.join(
            DATA_FOLDER, dataset_file_name[dataset_name]
        )

    def download_dataset(self):
        if os.path.exists(self.dataset_file_path):
            return

        response = requests.get(self.dataset_url)
        response.raise_for_status()
        decompressed_data = gzip.decompress(response.content)
        with open(self.dataset_file_path, "wb") as f_out:
            f_out.write(decompressed_data)

    def get_sparse_matrix(self):
        df = pd.read_csv(
            self.dataset_file_path,
            sep="\t",
            comment="#",
            header=None,
            names=["source", "target"],
        )
        num_nodes = max(df["source"].max(), df["target"].max()) + 1

        weights = np.ones(len(df), dtype=np.float32)
        adjacency_matrix = csr_array(
            (weights, (df["source"], df["target"])), shape=(num_nodes, num_nodes)
        )
        return adjacency_matrix

    def get_number_of_nodes(self):
        return dataset_number_unique_nodes[self.dataset_name]

    def get_number_of_edges(self):
        return dataset_number_edges[self.dataset_name]


data_managers = {name: SnapDataManager(name) for name in dataset_link.keys()}


def get_data_manager(dataset_name):
    assert dataset_name in data_managers, f"Dataset '{dataset_name}' not supported."
    return data_managers[dataset_name]
