import asyncio
from pagerank import pagerank, pagerank_sequential
import time
from data_manager import get_data_manager


async def main():
    dataset_name = "web-Google"
    data_manager = get_data_manager(dataset_name)
    data_manager.download_dataset()
    await data_manager.wait_for_download()
    adjacency_matrix = data_manager.get_sparse_matrix()
    num_nodes = adjacency_matrix.shape[0]

    print(
        f"Starting parallel PageRank computation for {dataset_name} with {num_nodes} nodes"
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


if __name__ == "__main__":
    asyncio.run(main())
