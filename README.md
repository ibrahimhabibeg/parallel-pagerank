# Parallel PageRank

<p align="center"><i>Parallel PageRank for Stanford SNAP web graphs</i></p>

This app is my submission for the course project for the course Parallel Programming (CSC 304) at the Suez Canal University in the spring semester of 2026.

The app implements the PageRank algorithm (Page et al., 1999) parallelly using Python's multiprocessing library. There is a streamlit interface that allows you to run the algorithm on one of the Stanford SNAP (Leskovec & Krevl, 2014) web graphs and compare the results with the sequential implementation.

You can access the app on [Hugging Face spaces](https://huggingface.co/spaces/ibrahimhabibeg/parallel-pagerank) or run it locally by following the instructions below. On Hugging Face not all the features are available since the app is running on a 2 cores CPU.

## The PageRank Algorithm

The algorithm is implemented as a Markov chain.

Given a graph with $n$ nodes, we represent the probability of being at each node at the timestep $t$ as a vector $\mathbf{v}_t \in \mathbb{R}^n$. We initialize $\mathbf{v}_0$ as a uniform distribution over the nodes, i.e., all values are $\frac{1}{n}$. Using the transition matrix $\mathbf{M} \in \mathbb{R}^{n \times n}$ (which will be defined below), we compute the next state of $\mathbf{v}$ as follows

$$
\mathbf{v}_{t+1} = \mathbf{M} \mathbf{v_t}
$$

Regarding the transition matrix $\mathbf{M}$ , $\mathbf{M_{ij}}$ is the probability of transitioning from node $j$ to node $i$. We define the damping factor $\alpha$ (usually set to 0.85) as the probability of transitioning to a neighboring node, and $1-\alpha$ as the probability of transitioning to any random node with equal probability. Given that the surfer will move to a neighboring node, all neighbors of node $j$ will have equal probability of being chosen (because we are assuming a non-weighted graph). If there are no neighbors for node $j$, then the surfer will transition to any random node with equal probability. Hence, we can write $\mathbf{M}_{ij}$ as follows

$$
\mathbf{M} = \alpha (\mathbf{A}\mathbf{D}^{-1} + \mathbf{E}) + (1-\alpha) \frac{1}{n} \mathbf{J}_{n}
$$

where $\mathbf{A}$ is the adjacency matrix of the graph, $\mathbf{D}$ is the diagonal matrix where $\mathbf{D_{jj}}$ is the out-degree of node $j$, $\mathbf{J_n}$ is the $n \times n$ matrix of all ones, and $\mathbf{E}$ is a matrix where $\mathbf{E_{ij}} = \frac{1}{n}$ if node $j$ has no neighbors and $0$ otherwise.

## Algorithm Implementation

$n$ is usually quite large; thus, we cannot possible store $\mathbf{M}$ in memory, e.g. for the web-Google graph $\mathbf{M}$ will need $2.8$ TB of memory using 32-bit floats. To overcome this issue, we rewrite the update step of $\mathbf{v}$ as follows

$$
\begin{aligned}
  \mathbf{v_{t+1}} &= \mathbf{M}\mathbf{v_t} \\
  &= \alpha \mathbf{A}\mathbf{D}^{-1} \mathbf{v_t} + \alpha \mathbf{E}\mathbf{v_t} + (1-\alpha) \frac{1}{n} \mathbf{J_n} \mathbf{v_t} \\
  &= \alpha \mathbf{A}\mathbf{D}^{-1} \mathbf{v_t} + \alpha \frac{1}{n} (\mathbf{e} \cdot \mathbf{v_t}) + (1-\alpha) \frac{1}{n} \\
\end{aligned}
$$

where $\mathbf{e}$ is an indicator vector of the nodes with no neighbors and broadcasting is used in the last step to add the two scalar values to each element of the vector.

$\mathbf{A}\mathbf{D}^{-1}$ is a sparse matrix; hence, we are able to store it in memory and multiply with it efficiently.

Regarding the parallel implementation, we parallelize the multiplication by splitting the matrix into blocks of rows and multiplying each block with $\mathbf{v_t}$ in a separate process. Finally, we merge all the results together.

We keep moving in the chain until the L2 norm of the difference between $\mathbf{v_{t+1}}$ and $\mathbf{v_t}$ is less than a certain threshold or we reach a maximum number of iterations.

## Running the App Locally

Follow the following steps:

Clone the repository

```bash
gh repo clone ibrahimhabibeg/parallel-pagerank
cd parallel-pagerank
```

Install the dependencies

```bash
uv sync
```

Run the app

```bash
uv run streamlit run app.py
```

## Acknowledgments

Special thanks to the course instructor Dr. Mohamed Khamis and the TA Eng. Amro Medhat for their guidance throughout the course.

## References

Albert, R., Jeong, H., & Barabási, A.-L. (1999). Diameter of the World-Wide Web. Nature, 401(6749), 130–131. https://doi.org/10.1038/43601

Google. (2002). Google Programming Contest.

Leskovec, J., & Krevl, A. (2014, June). SNAP Datasets: Stanford Large Network Dataset Collection. http://snap.stanford.edu/data

Leskovec, J., Lang, K. J., Dasgupta, A., & Mahoney, M. W. (2008). Community Structure in Large Networks: Natural Cluster Sizes and the Absence of Large Well-Defined Clusters. Internet Mathematics, 6, 123–129.

Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank Citation Ranking: Bringing Order to the Web. The Web Conference. https://api.semanticscholar.org/CorpusID:1508503
