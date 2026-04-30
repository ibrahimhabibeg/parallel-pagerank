import os
import streamlit as st
from data_manager import supported_datasets, get_data_manager
from threading import Thread
from pagerank import pagerank, pagerank_sequential
import time
from os import cpu_count
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_DATASET = supported_datasets[0]

is_on_huggingface = "SPACE_ID" in os.environ
number_of_processes = cpu_count() if not is_on_huggingface else 2
max_processors_to_use_in_graph = min(12, number_of_processes)


def init_state():
    st.session_state.setdefault("dataset", DEFAULT_DATASET)
    st.session_state.setdefault("damping_factor", 0.85)
    st.session_state.setdefault("show_algorithm_section", False)
    st.session_state.setdefault("is_running_algorithm", False)
    st.session_state.setdefault("are_values_computed", False)
    st.session_state.setdefault("computed_values", {})


def handle_dataset_change():
    st.session_state["dataset"] = st.session_state["dataset_selectbox"]
    st.session_state["show_algorithm_section"] = False
    st.session_state["is_running_algorithm"] = False
    st.session_state["are_values_computed"] = False
    st.session_state["computed_values"] = {}


def render_dataset_section():
    st.header("Step 1: Dataset Selection")
    st.write("Select a dataset to run PageRank on.")
    st.subheader("Available Datasets")
    st.write("""Four datasets from Stanford's [SNAP](https://snap.stanford.edu/data/index.html) (Leskovec & Krevl, 2014) project are available for selection:

- **web-Google**: Nodes represent web pages and directed edges represent hyperlinks between them. Released in 2002 by Google as a part of Google Programming Contest. (Leskovec et al., 2008; Google, 2002)
- **web-BerkStan**: Nodes represent pages from berkely.edu and stanford.edu domains and directed edges represent hyperlinks between them. (Leskovec et al., 2008)
- **web-Stanford**: Nodes represent pages from Stanford University (stanford.edu) and directed edges represent hyperlinks between them. (Leskovec et al., 2008)
- **web-NotreDame**: Nodes represent pages from University of Notre Dame (domain nd.edu) and directed edges represent hyperlinks between them. (Albert et al., 1999)
    """)

    st.selectbox(
        "Choose a dataset",
        options=supported_datasets,
        index=supported_datasets.index(DEFAULT_DATASET),
        disabled=st.session_state["is_running_algorithm"],
        on_change=handle_dataset_change,
        key="dataset_selectbox",
    )

    data_manager = get_data_manager(st.session_state["dataset"])
    num_nodes = data_manager.get_number_of_nodes()
    num_edges = data_manager.get_number_of_edges()
    col1, col2 = st.columns(2)
    col1.metric("Number of Nodes", f"{num_nodes:,}")
    col2.metric("Number of Edges", f"{num_edges:,}")


def handle_button_click():
    st.session_state["is_running_algorithm"] = True
    st.session_state["show_algorithm_section"] = True
    st.session_state["are_values_computed"] = False
    st.session_state["computed_values"] = {}


def handle_damping_factor_change():
    st.session_state["damping_factor"] = st.session_state["damping_factor_slider"]
    st.session_state["show_algorithm_section"] = False
    st.session_state["is_running_algorithm"] = False
    st.session_state["are_values_computed"] = False
    st.session_state["computed_values"] = {}


def render_algorithm_run_section():
    st.header("Step 2: Run PageRank")
    st.write("PageRank is used to rank web pages in a graph based on their importance.")
    st.write(
        "You can adjust the settings below and click the 'Run PageRank' button to execute the algorithm on the selected dataset."
    )
    st.write(
        "When you click 'Run PageRank', the algorithm will be run both sequentially and in parallel. There will also be a comparison between various values for the number of processes used."
    )

    with st.expander("Settings"):
        st.slider(
            "Damping Factor",
            min_value=0.1,
            max_value=0.99,
            value=st.session_state["damping_factor"],
            step=0.01,
            disabled=st.session_state["is_running_algorithm"],
            on_change=handle_damping_factor_change,
            key="damping_factor_slider",
        )

    st.button(
        "Run PageRank",
        type="primary",
        width="stretch",
        disabled=st.session_state["is_running_algorithm"],
        on_click=handle_button_click,
    )


def render_results_section_from_computed_values():
    computed_values = st.session_state["computed_values"]
    parallel_results = computed_values["parallel_results"]
    sequential_results = computed_values["sequential_results"]
    parallel_running_time = computed_values["parallel_running_time"]
    sequential_running_time = computed_values["sequential_running_time"]
    all_process_counts = computed_values["all_process_counts"]
    all_running_times = computed_values["all_running_times"]

    cols = st.columns(2)
    cols[0].subheader("Parallel PageRank")
    cols[1].subheader("Sequential PageRank")

    with cols[0]:
        st.metric(
            "Running Time (seconds)",
            f"{parallel_running_time:.2f} seconds",
            delta=f"{parallel_running_time - sequential_running_time:.2f} seconds",
            delta_color="inverse",
        )
    with cols[1]:
        st.metric(
            "Running Time (seconds)",
            f"{sequential_running_time:.2f} seconds",
            delta=f"{sequential_running_time - parallel_running_time:.2f} seconds",
            delta_color="inverse",
        )

    cols[0].metric("Speedup", f"{sequential_running_time / parallel_running_time:.2f}x")
    cols[1].metric(
        "Efficiency",
        f"{(sequential_running_time / parallel_running_time) / number_of_processes:.2f}",
    )

    if is_on_huggingface:
        render_hf_warning()
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=all_process_counts,
            y=[sequential_running_time / t for t in all_running_times],
            name="Speedup",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=all_process_counts,
            y=[
                (sequential_running_time / t) / p
                for t, p in zip(all_running_times, all_process_counts)
            ],
            name="Efficiency",
        ),
        secondary_y=True,
    )
    fig.update_layout(title_text="Speedup and Efficiency vs Number of Processes")
    fig.update_yaxes(title_text="Speedup", secondary_y=False)
    fig.update_yaxes(title_text="Efficiency", secondary_y=True)

    st.plotly_chart(fig, width="stretch")


def render_hf_warning():
    st.warning("""
    This app is running on Hugging Face Spaces and has limited resources.
    There is only 2 CPU cores available, so no comparison between different number of processes can be made.
    Try running the app locally to see the full experience and compare the performance of different number of processes.
    """)


def render_results_section():
    st.header("Step 3: View Results")

    if st.session_state["are_values_computed"]:
        render_results_section_from_computed_values()
        return

    st.write("After running PageRank, the results will be displayed here.")

    st.info(f"Running with **{number_of_processes}** processes for parallel PageRank.")

    parallel_results = None
    sequential_results = None
    parallel_running_time = None
    sequential_running_time = None
    adjacency_matrix = get_data_manager(st.session_state["dataset"]).get_sparse_matrix()

    cols = st.columns(2)
    cols[0].subheader("Parallel PageRank")
    cols[1].subheader("Sequential PageRank")
    parallel_metrics_placeholder = cols[0].empty()
    sequential_metrics_placeholder = cols[1].empty()

    with cols[0]:
        with st.spinner("Wait for it...", show_time=True):
            start_time = time.time()
            parallel_results = pagerank(
                adjacency_matrix,
                alpha=st.session_state["damping_factor"],
                number_of_workers=number_of_processes,
            )
            end_time = time.time()
            parallel_running_time = end_time - start_time
        parallel_metrics_placeholder.metric(
            "Running Time (seconds)", f"{parallel_running_time:.2f} seconds"
        )
    with cols[1]:
        with st.spinner("Wait for it...", show_time=True):
            start_time = time.time()
            sequential_results = pagerank_sequential(
                adjacency_matrix,
                alpha=st.session_state["damping_factor"],
            )
            end_time = time.time()
            sequential_running_time = end_time - start_time
        sequential_metrics_placeholder.metric(
            "Running Time (seconds)", f"{sequential_running_time:.2f} seconds"
        )

    parallel_metrics_placeholder.metric(
        "Running Time (seconds)",
        f"{parallel_running_time:.2f} seconds",
        delta=f"{parallel_running_time - sequential_running_time:.2f} seconds",
        delta_color="inverse",
    )
    sequential_metrics_placeholder.metric(
        "Running Time (seconds)",
        f"{sequential_running_time:.2f} seconds",
        delta=f"{sequential_running_time - parallel_running_time:.2f} seconds",
        delta_color="inverse",
    )

    cols[0].metric("Speedup", f"{sequential_running_time / parallel_running_time:.2f}x")
    cols[1].metric(
        "Efficiency",
        f"{(sequential_running_time / parallel_running_time) / number_of_processes:.2f}",
    )

    if is_on_huggingface:
        render_hf_warning()
        st.session_state["computed_values"] = {
            "parallel_results": parallel_results,
            "sequential_results": sequential_results,
            "parallel_running_time": parallel_running_time,
            "sequential_running_time": sequential_running_time,
            "all_process_counts": [number_of_processes],
            "all_running_times": [parallel_running_time],
        }
        st.session_state["is_running_algorithm"] = False
        st.session_state["are_values_computed"] = True
        st.rerun()
        return

    progress_info = st.info(
        "Running the algorithm with different number of processes to show the effect on running time..."
    )
    progress_bar = st.progress(1)
    chart_placeholder = st.empty()
    all_efficiencies = []
    all_speedups = []
    all_process_counts = []
    all_running_times = []

    for num_processes in range(2, min(12, number_of_processes) + 1, 2):
        pbar_num_steps = (min(12, number_of_processes) - 2) // 2 + 1
        start_time = time.time()
        pagerank(
            adjacency_matrix,
            alpha=st.session_state["damping_factor"],
            number_of_workers=num_processes,
        )
        end_time = time.time()
        running_time = end_time - start_time
        progress_bar.progress((num_processes / 2) / pbar_num_steps)
        all_process_counts.append(num_processes)
        all_running_times.append(running_time)
        all_speedups.append(sequential_running_time / running_time)
        all_efficiencies.append(
            (sequential_running_time / running_time) / num_processes
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=all_process_counts,
                y=all_speedups,
                name="Speedup",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=all_process_counts,
                y=all_efficiencies,
                name="Efficiency",
            ),
            secondary_y=True,
        )
        fig.update_layout(title_text="Speedup and Efficiency vs Number of Processes")
        fig.update_yaxes(title_text="Speedup", secondary_y=False)
        fig.update_yaxes(title_text="Efficiency", secondary_y=True)

        chart_placeholder.plotly_chart(fig, width="stretch")

    progress_info.empty()
    progress_bar.empty()

    st.session_state["computed_values"] = {
        "parallel_results": parallel_results,
        "sequential_results": sequential_results,
        "parallel_running_time": parallel_running_time,
        "sequential_running_time": sequential_running_time,
        "all_process_counts": all_process_counts,
        "all_running_times": all_running_times,
    }
    st.session_state["is_running_algorithm"] = False
    st.session_state["are_values_computed"] = True
    st.rerun()


def render_info_section():
    st.title("Parallel PageRank")

    st.write("""
    This app is the submission for the course project for the course Parallel Programming (CSC 304) at the Suez Canal University in the spring semester of 2026.
    
    This app implements a $O(|V| + |E|)$ algorithm for computing the PageRank (Page et al., 1999) of a graph both sequentially and in parallel using Python's multiprocessing module.
    The app compares the running time of the parallel and sequential implementations and shows the speedup and efficiency of the parallel implementation for different number of processes.
    
    This GUI is built using Streamlit and allows the user to run the algorithm on the Web graphs datasets from Stanford's SNAP project (Leskovec & Krevl, 2014).
    
    The code can be found in the repo [ibrahimhabibeg/parallel-pagerank](https://github.com/ibrahimhabibeg/parallel-pagerank)
    """)

    columns = st.columns(2)
    with columns[0]:
        st.metric("Developer", "[Ibrahim Habib](https://ibrahimhabib.me)")
        st.metric(
            "GitHub Repository",
            "[parallel-pagerank](https://github.com/ibrahimhabibeg/parallel-pagerank)",
        )
    with columns[1]:
        st.metric("Course Instructor", "Dr. Mohamed Khamis")
        st.metric("Course TA", "Eng. Amro Medhat")


def render_refrences_section():
    st.header("References")
    st.write("""
Albert, R., Jeong, H., & Barabási, A.-L. (1999). Diameter of the World-Wide Web. Nature, 401(6749), 130–131. https://doi.org/10.1038/43601

Google. (2002). Google Programming Contest.

Leskovec, J., & Krevl, A. (2014, June). SNAP Datasets: Stanford Large Network Dataset Collection. http://snap.stanford.edu/data

Leskovec, J., Lang, K. J., Dasgupta, A., & Mahoney, M. W. (2008). Community Structure in Large Networks: Natural Cluster Sizes and the Absence of Large Well-Defined Clusters. Internet Mathematics, 6, 123–129.

Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank Citation Ranking: Bringing Order to the Web. The Web Conference. https://api.semanticscholar.org/CorpusID:1508503
    """)


def run_streamlit_app():
    init_state()

    st.set_page_config(page_title="Parallel PageRank")
    render_info_section()
    st.divider()
    render_dataset_section()
    st.divider()
    render_algorithm_run_section()
    st.divider()
    if st.session_state["show_algorithm_section"]:
        render_results_section()
        st.divider()
    render_refrences_section()


def download_datasets():
    data_mangers = [get_data_manager(name) for name in supported_datasets]
    download_threads = [
        Thread(target=data_manager.download_dataset) for data_manager in data_mangers
    ]
    for thread in download_threads:
        thread.start()
    for thread in download_threads:
        thread.join()


if __name__ == "__main__":
    download_datasets()
    run_streamlit_app()
