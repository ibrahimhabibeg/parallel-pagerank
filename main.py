import pandas as pd
import numpy as np
from scipy.sparse import csr_array
from pagerank import pagerank

if __name__ == "__main__":
    path_to_file = "data/web-Google.txt"

    df = pd.read_csv(
        path_to_file, sep="\t", comment="#", header=None, names=["source", "target"]
    )
    num_nodes = max(df["source"].max(), df["target"].max()) + 1

    weights = np.ones(len(df), dtype=np.float32)
    adjacency_matrix = csr_array(
        (weights, (df["source"], df["target"])), shape=(num_nodes, num_nodes)
    )
    probs = pagerank(adjacency_matrix)
    print(probs[:10])
    print(probs.sum())
