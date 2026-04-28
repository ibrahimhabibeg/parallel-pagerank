import numpy as np
from multiprocessing import Process, cpu_count, Queue


def pagerank_worker(M, alpha, start_row, end_row, input_queue, output_queue):
    while True:
        v = input_queue.get()

        if v is None:
            break

        K, N = M.shape
        new_v = np.zeros(K)

        for i in range(K):
            row_start = M.indptr[i]
            row_end = M.indptr[i + 1]
            if row_start < row_end:
                row_indices = M.indices[row_start:row_end]
                row_data = M.data[row_start:row_end]
                new_v[i] = np.sum(row_data * v[row_indices])

        new_v = alpha * new_v + (1 - alpha) / N
        output_queue.put((new_v, start_row, end_row))


def create_markov_matrix(A):
    A = A.tocoo()
    col_sums = np.array(A.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1
    A.data = A.data / col_sums[A.col]
    return A.tocsr()


def pagerank(A, alpha=0.85, max_iter=100, tol=1e-6, number_of_workers=cpu_count()):
    out_degrees = np.array(A.sum(axis=0)).flatten()
    deadends = np.where(out_degrees == 0)[0]

    M = create_markov_matrix(A)
    N = M.shape[0]
    K = N // number_of_workers
    m = N % number_of_workers
    row_starts = [i * K + min(i, m) for i in range(number_of_workers)]
    row_ends = [(i + 1) * K + min(i + 1, m) for i in range(number_of_workers)]

    input_queues = [Queue() for _ in range(number_of_workers)]
    output_queue = Queue()

    v = np.ones(N) / N
    print(f"Sum of initial v: {v.sum():.6f}")

    workers = [
        Process(
            target=pagerank_worker,
            args=(
                M[row_starts[i] : row_ends[i]],
                alpha,
                row_starts[i],
                row_ends[i],
                input_queues[i],
                output_queue,
            ),
        )
        for i in range(number_of_workers)
    ]

    for w in workers:
        w.start()

    for _ in range(max_iter):
        for q in input_queues:
            q.put(v)

        new_v = np.zeros(N)
        for _ in range(number_of_workers):
            worker_v, start_row, end_row = output_queue.get()
            new_v[start_row:end_row] = worker_v

        deadends_prob = v[deadends].sum()
        new_v += alpha * deadends_prob / N

        diff_norm = np.sqrt(np.sum((new_v - v) ** 2))

        if diff_norm < tol:
            break

        v = new_v

    for q in input_queues:
        q.put(None)
    for w in workers:
        w.join()

    return v


def pagerank_sequential_step(M, v, alpha):
    N = M.shape[0]
    new_v = np.zeros(N)
    for i in range(N):
        row_start = M.indptr[i]
        row_end = M.indptr[i + 1]
        if row_start < row_end:
            row_indices = M.indices[row_start:row_end]
            row_data = M.data[row_start:row_end]
            new_v[i] = np.sum(row_data * v[row_indices])
    new_v = alpha * new_v + (1 - alpha) / N
    return new_v


def pagerank_sequential(A, alpha=0.85, max_iter=100, tol=1e-6):
    out_degrees = np.array(A.sum(axis=0)).flatten()
    deadends = np.where(out_degrees == 0)[0]

    M = create_markov_matrix(A)
    N = M.shape[0]
    v = np.ones(N) / N
    for _ in range(max_iter):
        new_v = pagerank_sequential_step(M, v, alpha)
        new_v += alpha * v[deadends].sum() / N
        
        diff_norm = np.sqrt(np.sum((new_v - v) ** 2))
        if diff_norm < tol:
            break
        
        v = new_v
    return v
