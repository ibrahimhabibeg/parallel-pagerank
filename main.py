import pandas as pd
import numpy as np
from scipy.sparse import csr_array
from pagerank import pagerank, pagerank_sequential
import time

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
    print(
        f"Starting parallel PageRank computation for {num_nodes} nodes and {len(df)} edges..."
    )
    time_start = time.time()
    probs = pagerank(adjacency_matrix)
    time_end = time.time()
    print(probs.sum())
    print(f"Parallel PageRank completed in {time_end - time_start:.4f}, seconds.")
    print("Starting sequential PageRank computation...")
    time_start_seq = time.time()
    probs_seq = pagerank_sequential(adjacency_matrix)
    time_end_seq = time.time()
    print(probs_seq.sum())
    print(
        f"Sequential PageRank completed in {time_end_seq - time_start_seq:.4f}, seconds."
    )
