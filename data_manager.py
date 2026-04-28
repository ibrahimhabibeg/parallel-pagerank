import asyncio
import os
import aiohttp
import aiofiles
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

DATA_FOLDER = "data"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

class SnapDataManager:
    def __init__(self, dataset_name):
        assert dataset_name in dataset_link, f"Dataset '{dataset_name}' not supported."
        self.dataset_name = dataset_name
        self.dataset_url = dataset_link[dataset_name]
        self.dataset_file_path = os.path.join(DATA_FOLDER, dataset_file_name[dataset_name])
        self.download_thread = None


    async def _download_dataset_task(self):
        if os.path.exists(self.dataset_file_path):
            return

        async with aiohttp.ClientSession() as session:
            async with session.get(self.dataset_url) as response:
                response.raise_for_status()
                compressed_data = await response.read()
                decompressed_data = gzip.decompress(compressed_data)
                async with aiofiles.open(self.dataset_file_path, "wb") as f_out:
                    await f_out.write(decompressed_data)

    def download_dataset(self):
        if self.download_thread is None or not self.download_thread.done():
            self.download_thread = asyncio.create_task(self._download_dataset_task())

    async def wait_for_download(self):
        if self.download_thread is not None:
            await self.download_thread

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


data_managers = {name: SnapDataManager(name) for name in dataset_link.keys()}


def get_data_manager(dataset_name):
    assert dataset_name in data_managers, f"Dataset '{dataset_name}' not supported."
    return data_managers[dataset_name]
