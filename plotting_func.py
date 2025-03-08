import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
from matplotlib import gridspec
from dataclasses import dataclass
import sklearn.datasets
from tensorflow.keras.datasets import mnist


@dataclass
class SwissRoll:
    datapoints: np.ndarray
    labels: np.ndarray
    n_samples: int

    @classmethod
    def generate(self, n_samples, noise):
        datapoints, labels = sklearn.datasets.make_swiss_roll(
            n_samples=n_samples, noise=noise
        )
        return SwissRoll(datapoints=datapoints, labels=labels, n_samples=n_samples)

    def plot(self, width=None, height=None, title=None):
        plot_swiss_roll_plotly(
            self.datapoints,
            self.labels,
            n_samples=self.n_samples,
            width=width,
            height=height,
            title=title,
        )


@dataclass
class MNIST:
    data_train: np.array
    labels_train: np.array
    data_test: np.array
    labels_test: np.array

    @classmethod
    def generate(self, n_training_samples):
        (data_train, labels_train), (data_test, labels_test) = mnist.load_data()
        data_train = data_train[:n_training_samples]
        labels_train = labels_train[:n_training_samples]
        return MNIST(data_train, labels_train, data_test, labels_test)

    def reshape(self):
        self.data_train = self.data_train.reshape(self.data_train.shape[0], -1)
        self.data_test = self.data_test.reshape(self.data_test.shape[0], -1)


@dataclass
class TSNEResult:
    """Dataclass to store the results of a single t-SNE algorithm run."""

    dataset_name: str
    n_samples: int
    n_iter: int
    embedding: np.ndarray
    optimization_mode: str | None
    kl_divergence: float
    initial_alpha: float
    alpha_lr: float | None
    im_embeddings: list
    im_KLs: np.ndarray
    im_alphas: np.ndarray
    im_alpha_grads: np.ndarray


@dataclass
class TSNEResultsWithKNN(TSNEResult):
    """Dataclass to store the results of a single t-SNE algorithm run with KNN affinities."""

    knn_recall: float


def compute_knn_recall(
    original_data: np.ndarray, tsne_data: np.ndarray, k: int = 10
) -> float:
    """
    Computes the recall of k-nearest neighbors between the original data and the t-SNE data.

    Parameters
    ----------
    original_data : np.ndarray
        The original multidimensional data.
    tsne_data : np.ndarray
        The t-SNE transformed data.
    k : int, optional
        The number of neighbors to consider, by default 7

    Returns
    -------
    float
        The average recall of k-nearest neighbors between the original data and the t-SNE data.

    Notes
    -----

    The formula is taken from: Gove et al. (2022)
    New guidance for using t-SNE: Alternative defaults, hyperparameter selection automation,
    and comparative evaluation,
    Visual Informatics, Volume 6, Issue 2, 2022,

    """
    # Fit kNN on original data
    knn_orig = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn_orig.fit(original_data)
    orig_neighbors = knn_orig.kneighbors(return_distance=False)

    # Fit kNN on t-SNE data
    knn_tsne = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn_tsne.fit(tsne_data)
    tsne_neighbors = knn_tsne.kneighbors(return_distance=False)

    # Calculate recall for each point
    recall_scores = np.zeros(len(original_data))
    for i in range(len(original_data)):
        shared_neighbors = np.intersect1d(orig_neighbors[i], tsne_neighbors[i])
        recall = len(shared_neighbors) / k
        recall_scores[i] = recall
    # Return average recall
    return np.mean(recall_scores)


### PLOTTING FUNCTIONS ###


def plot_tsne_result(
    data: TSNEResult | TSNEResultsWithKNN,
    labels: np.ndarray,
    additional_title: str = "",
    marker_size: int = 3,
):
    fig = plt.figure(figsize=(10, 8))
    # fig.patch.set_facecolor("lightgrey")

    # set up axes
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.4])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    # ax1.set_facecolor("lightgrey")
    # ax2.set_facecolor("lightgrey")
    # ax3.set_facecolor("lightgrey")
    # ax4.set_facecolor("lightgrey")

    ax1.scatter(
        data.embedding[:, 0],
        data.embedding[:, 1],
        c=labels,
        s=marker_size,
        cmap="jet",
    )
    ax1.set_xticks([])
    ax1.set_yticks([])
    sns.despine(bottom=True, left=True, ax=ax1)

    ax2.plot(data.im_KLs, color="blue")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("KL divergence", color="blue")
    sns.despine(bottom=False, left=False, ax=ax2)

    ax3.plot(data.im_alphas, color="red")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("alpha", color="red")
    sns.despine(bottom=False, left=False, ax=ax3)

    ax4.plot(data.im_alpha_grads, color="green", label="alpha gradient")
    ax4.hlines(0, 0, data.n_iter, color="red", linestyle="--", label="zero gradient")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("alpha gradient", color="green")
    sns.despine(bottom=False, left=False, ax=ax4)

    if hasattr(data, "knn_recall"):
        knn_recall = data.knn_recall
        knn_recall = f"{knn_recall:.2f}"
    else:
        knn_recall = "N/A"

    if data.optimization_mode:
        title_param_1 = "adaptive"
        title_param_2 = f"Initial alpha = {data.initial_alpha}. Alpha learning rate = {data.alpha_lr}"
        title_param_3 = f"Final alpha = {data.im_alphas[-1]:.2f}. Final Loss = {data.kl_divergence:.2f}. kNN recall = {knn_recall}"
    else:
        title_param_1 = "fixed"
        title_param_2 = f"Alpha = {data.initial_alpha}. Final Loss = {data.kl_divergence:.2f}. kNN recall = {knn_recall}"
        title_param_3 = ""

    plt.suptitle(
        f"t-SNE embedding with {title_param_1} degrees of freedom\n{data.dataset_name} with {data.n_samples} samples\n{title_param_2}\n{title_param_3}\n{additional_title}",
        fontsize=12,
        fontweight="bold",
        color="black",
    )

    plt.tight_layout()
    plt.savefig(
        f"Figures/{data.dataset_name}_{data.n_samples}_samples (matplotlib).pdf"
    )

    plt.show()


def plot_side_by_side(
    data1: TSNEResult | TSNEResultsWithKNN,
    data2: TSNEResult | TSNEResultsWithKNN,
    labels: np.ndarray,
    additional_title_1: str = "",
    additional_title_2: str = "",
    marker_size: int = 3,
):
    fig = plt.figure(figsize=(16, 8))

    # set up axes
    gs = gridspec.GridSpec(2, 6, height_ratios=[1, 0.4])

    # plot the first embedding

    ax1_1 = fig.add_subplot(gs[0, :3])
    ax1_2 = fig.add_subplot(gs[1, 0])
    ax1_3 = fig.add_subplot(gs[1, 1])
    ax1_4 = fig.add_subplot(gs[1, 2])

    ax1_1.scatter(data1.embedding[:, 0], data1.embedding[:, 1], c=labels, s=marker_size)
    ax1_1.set_xticks([])
    ax1_1.set_yticks([])

    if hasattr(data1, "knn_recall"):
        knn_recall_data1 = data1.knn_recall
        knn_recall_data1 = f"{knn_recall_data1:.2f}"
    else:
        knn_recall_data1 = "N/A"

    if hasattr(data2, "knn_recall"):
        knn_recall_data2 = data2.knn_recall
        knn_recall_data2 = f"{knn_recall_data2:.2f}"
    else:
        knn_recall_data2 = "N/A"

    if data1.optimization_mode:
        title_param_1 = "adaptive"
        title_param_2 = f"Initial alpha = {data1.initial_alpha}. Alpha learning rate = {data1.alpha_lr}"
        title_param_3 = f"Final alpha = {data1.im_alphas[-1]:.2f}. Final Loss = {data1.kl_divergence:.2f}. kNN recall = {knn_recall_data1}"
    else:
        title_param_1 = "fixed"
        title_param_2 = f"Alpha = {data1.initial_alpha}. Final Loss = {data1.kl_divergence:.2f}. kNN recall = {knn_recall_data1}"
        title_param_3 = ""

    ax1_1.set_title(
        f"t-SNE embedding with {title_param_1} degrees of freedom\n{data1.dataset_name} with {data1.n_samples} samples\n{title_param_2}\n{title_param_3}\n{additional_title_1}",
        fontsize=12,
        fontweight="bold",
        color="black",
    )

    sns.despine(bottom=True, left=True, ax=ax1_1)

    ax1_2.plot(data1.im_KLs, color="blue")
    ax1_2.set_xlabel("Iteration")
    ax1_2.set_ylabel("KL divergence", color="blue")
    sns.despine(bottom=False, left=False, ax=ax1_2)

    ax1_3.plot(data1.im_alphas, color="red")
    ax1_3.set_xlabel("Iteration")
    ax1_3.set_ylabel("alpha", color="red")
    sns.despine(bottom=False, left=False, ax=ax1_3)

    ax1_4.plot(data1.im_alpha_grads, color="green", label="alpha gradient")
    ax1_4.hlines(0, 0, data1.n_iter, color="red", linestyle="--", label="zero gradient")
    ax1_4.set_xlabel("Iteration")
    ax1_4.set_ylabel("alpha gradient", color="green")
    sns.despine(bottom=False, left=False, ax=ax1_4)

    # plot the second embedding

    ax2_1 = fig.add_subplot(gs[0, 3:])
    ax2_2 = fig.add_subplot(gs[1, 3])
    ax2_3 = fig.add_subplot(gs[1, 4])
    ax2_4 = fig.add_subplot(gs[1, 5])

    ax2_1.scatter(data2.embedding[:, 0], data2.embedding[:, 1], c=labels, s=marker_size)
    ax2_1.set_xticks([])
    ax2_1.set_yticks([])

    if data2.optimization_mode:
        title_param_1 = "adaptive"
        title_param_2 = f"Initial alpha = {data2.initial_alpha}. Alpha learning rate = {data2.alpha_lr}"
        title_param_3 = f"Final alpha = {data2.im_alphas[-1]:.2f}. Final Loss = {data2.kl_divergence:.2f}. kNN recall = {knn_recall_data2}"
    else:
        title_param_1 = "fixed"
        title_param_2 = f"Alpha = {data2.initial_alpha}. Final Loss = {data2.kl_divergence:.2f}. kNN recall = {knn_recall_data2}"
        title_param_3 = ""

    ax2_1.set_title(
        f"t-SNE embedding with {title_param_1} degrees of freedom\n{data2.dataset_name} with {data2.n_samples} samples\n{title_param_2}\n{title_param_3}\n{additional_title_2}",
        fontsize=12,
        fontweight="bold",
        color="black",
    )

    sns.despine(bottom=True, left=True, ax=ax2_1)

    ax2_2.plot(data2.im_KLs, color="blue")
    ax2_2.set_xlabel("Iteration")
    ax2_2.set_ylabel("KL divergence", color="blue")
    sns.despine(bottom=False, left=False, ax=ax2_2)

    ax2_3.plot(data2.im_alphas, color="red")
    ax2_3.set_xlabel("Iteration")
    ax2_3.set_ylabel("alpha", color="red")
    sns.despine(bottom=False, left=False, ax=ax2_3)

    ax2_4.plot(data2.im_alpha_grads, color="green", label="alpha gradient")
    ax2_4.hlines(0, 0, data2.n_iter, color="red", linestyle="--", label="zero gradient")
    ax2_4.set_xlabel("Iteration")
    ax2_4.set_ylabel("alpha gradient", color="green")
    sns.despine(bottom=False, left=False, ax=ax2_4)

    plt.tight_layout()

    if data1.dataset_name == data2.dataset_name and data1.n_samples == data2.n_samples:
        fig_title = f"Comparison of t-SNE embeddings for {data1.dataset_name} with {data1.n_samples} samples"
    else:
        fig_title = f"Comparison of t-SNE embeddings"

    plt.savefig(f"Figures/{fig_title} (matplotlib).pdf")
    plt.show()


def plot_tsne_result_plotly(
    data: TSNEResult | TSNEResultsWithKNN,
    labels: np.ndarray,
    additional_title: str = "",
    marker_size: int = 3,
    black_template: bool = False,
):
    if black_template:
        template = "plotly_dark"
        text_color = "white"
    else:
        template = "plotly"
        text_color = "black"

    # Create subplot layout
    fig = sp.make_subplots(
        rows=2,
        cols=3,
        specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
        subplot_titles=["", "KL Divergence", "Alpha Values", "Alpha Gradient"],
        vertical_spacing=0.05,
        horizontal_spacing=0.1,
        row_heights=[0.7, 0.3],
    )

    # t-SNE Embedding Plot
    fig.add_trace(
        go.Scatter(
            x=data.embedding[:, 0],
            y=data.embedding[:, 1],
            mode="markers",
            marker=dict(
                color=labels,
                size=marker_size,
                showscale=False,
                colorscale="Jet",
            ),
            name="t-SNE Embedding",
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)

    # KL Divergence Plot
    min_kl = min(data.im_KLs)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(data.im_KLs))),
            y=data.im_KLs,
            mode="lines",
            line=dict(color="blue"),
            name="KL Divergence",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=min_kl,
        line_dash="dash",
        line_color="orange" if black_template else "black",
        annotation_text="min 𝓛",
        annotation_position="top right",
        opacity=0.8,
        row=2,
        col=1,
    )
    fig.update_yaxes(tickvals=[min_kl], ticktext=[f"{min_kl:.2f}"], row=2, col=1)

    # Alpha Values Plot
    last_alpha = data.im_alphas[-1]
    fig.add_trace(
        go.Scatter(
            x=list(range(len(data.im_alphas))),
            y=data.im_alphas,
            mode="lines",
            line=dict(color="red"),
            name="Alpha Values",
        ),
        row=2,
        col=2,
    )
    fig.add_hline(
        y=last_alpha,
        line_dash="dash",
        line_color="green" if black_template else "black",
        annotation_text="Converged α",
        annotation_position="bottom right",
        opacity=0.8,
        row=2,
        col=2,
    )
    fig.update_yaxes(
        tickvals=[last_alpha], ticktext=[f"{last_alpha:.2f}"], row=2, col=2
    )

    # Alpha Gradient Plot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(data.im_alpha_grads))),
            y=data.im_alpha_grads,
            mode="lines",
            line=dict(color="green"),
            name="Alpha Gradient",
        ),
        row=2,
        col=3,
    )

    fig.update_yaxes(tickvals=[0], ticktext=["0"], row=2, col=3)

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red" if black_template else "black",
        annotation_text="Zero Gradient",
        annotation_position="bottom right",
        opacity=0.8,
        row=2,
        col=3,
    )

    for i in range(1, 4):
        fig.update_xaxes(
            tickvals=[0, 250, 500],
            ticktext=["0", "250", "500"],
            range=[0, 500],
            row=2,
            col=i,
        )

    # Titles and Layout
    fig.update_annotations(
        font=dict(size=16, color=text_color, family="Courier New, monospace")
    )
    axis_label_style = dict(size=16, color=text_color, family="Courier New, monospace")
    fig.update_xaxes(title=dict(text="Iterations", font=axis_label_style), row=2, col=2)
    knn_recall = getattr(data, "knn_recall", "N/A")
    title_param_1 = "adaptive" if data.optimization_mode else "fixed"
    title_param_2 = (
        f"Initial α = {data.initial_alpha}, α learning rate = {data.alpha_lr}"
        if data.optimization_mode
        else f"α = {data.initial_alpha}"
    )
    title_param_3 = f"Final α = {data.im_alphas[-1]:.2f}, Final 𝓛 = {data.kl_divergence:.2f}, kNN recall = {knn_recall:.2f}"

    fig_title = (
        f"t-SNE embedding with {title_param_1} α parameter<br>"
        f"{data.dataset_name} with {data.n_samples} samples<br>"
    )

    fig.update_layout(
        title=dict(
            text=(
                f"{fig_title}{title_param_2}<br>{title_param_3}<br>{additional_title}"
            ),
            x=0.5,
            y=0.95,
            yanchor="top",
            font=dict(size=16, color=text_color, family="Courier New, monospace"),
        ),
        template=template,
        height=800,
        width=1000,
        showlegend=False,
    )

    fig.write_image(f"figures/{fig_title} (plotly).pdf")
    fig.show()


def plot_side_by_side_plotly(
    data1: TSNEResult | TSNEResultsWithKNN,
    data2: TSNEResult | TSNEResultsWithKNN,
    labels: np.ndarray,
    additional_title_1: str = "",
    additional_title_2: str = "",
    marker_size: int = 3,
    title="",
):
    # Extract titles for the embeddings
    def get_titles(data, knn_recall, additional_title):
        if data.optimization_mode:
            mode = "adaptive"
            params = f"Initial α = {data.initial_alpha:.2f}, α learning rate = {data.alpha_lr}"
            summary = (
                f"Final α = {data.im_alphas[-1]:.2f}, "
                f"Final Loss = {data.kl_divergence:.2f}, "
                f"kNN recall = {knn_recall:.2f}"
            )
        else:
            mode = "fixed"
            params = f"α = {data.initial_alpha:.2f}"
            summary = (
                f"Final Loss = {data.kl_divergence:.2f}, kNN recall = {knn_recall:.2f}"
            )
        # insert empty string

        return f"t-SNE embedding with {mode} α parameter<br>{params}<br>{summary}<br>{additional_title}"

    knn_recall_data1 = getattr(data1, "knn_recall", "N/A")
    knn_recall_data2 = getattr(data2, "knn_recall", "N/A")

    title1 = get_titles(data1, knn_recall_data1, additional_title_1)
    title2 = get_titles(data2, knn_recall_data2, additional_title_2)

    axis_label_style = dict(size=20, color="white", family="Courier New, monospace")

    # Create subplot layout
    fig = sp.make_subplots(
        rows=2,
        cols=6,
        specs=[
            [{"colspan": 3}, None, None, {"colspan": 3}, None, None],
            [{}, {}, {}, {}, {}, {}],
        ],
        subplot_titles=[
            title1,
            title2,
            "",
            "",
            "",
            "",
        ],
        vertical_spacing=0.02,
        horizontal_spacing=0.05,
        row_heights=[0.7, 0.3],
        # lower the subtitles
    )

    # Final iteration for x-axis
    x_ticks = [0, 250, 500]

    def add_embedding(data, col, row, labels):
        fig.add_trace(
            go.Scatter(
                x=data.embedding[:, 0],
                y=data.embedding[:, 1],
                mode="markers",
                marker=dict(
                    color=labels,
                    size=marker_size,
                    showscale=False,
                    colorscale="Rainbow",
                ),
                name=f"t-SNE Embedding: {data.dataset_name}",
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(
            showgrid=False, zeroline=False, showticklabels=False, row=row, col=col
        )
        fig.update_yaxes(
            showgrid=False, zeroline=False, showticklabels=False, row=row, col=col
        )

    def add_metric_plot(
        data, col, row, metric, name, line_color, hline=None, hline_color=None
    ):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(metric))),
                y=metric,
                mode="lines",
                line=dict(color=line_color),
                name=name,
            ),
            row=row,
            col=col,
        )
        if hline is not None:
            fig.add_hline(
                y=hline,
                line_dash="dash",
                line_color=hline_color if hline_color else line_color,
                annotation_text=name,
                annotation_position="bottom right" if hline == 0 else "top right",
                row=row,
                col=col,
            )
        fig.update_xaxes(
            tickvals=x_ticks,
            ticktext=[str(tick) for tick in x_ticks],
            range=[0, 500],
            title=dict(
                text="Iterations", font=dict(size=24, family="Courier New, monospace")
            )
            if col == 2 or col == 5
            else None,
            row=row,
            col=col,
        )

        fig.update_yaxes(row=row, col=col, tickvals=[hline], ticktext=[f"{hline:.2f}"])

    # Add first embedding and metrics
    add_embedding(data1, 1, 1, labels)
    add_metric_plot(
        data1, 1, 2, data1.im_KLs, "min 𝓛", "blue", min(data1.im_KLs), "orange"
    )
    add_metric_plot(
        data1, 2, 2, data1.im_alphas, "Converged α", "red", data1.im_alphas[-1], "green"
    )
    add_metric_plot(
        data1, 3, 2, data1.im_alpha_grads, "Zero Gradient", "green", 0, "red"
    )

    # Add second embedding and metrics
    add_embedding(data2, 4, 1, labels)
    add_metric_plot(
        data2, 4, 2, data2.im_KLs, "min 𝓛", "blue", min(data2.im_KLs), "orange"
    )
    add_metric_plot(
        data2, 5, 2, data2.im_alphas, "Converged α", "red", data2.im_alphas[-1], "green"
    )
    add_metric_plot(
        data2, 6, 2, data2.im_alpha_grads, "Zero Gradient", "green", 0, "red"
    )

    # Generate title if embeddings are the same
    if data1.dataset_name == data2.dataset_name and data1.n_samples == data2.n_samples:
        fig_title = f"Comparison of t-SNE embeddings for {data1.dataset_name} with {data1.n_samples} samples"
    else:
        fig_title = f"Comparison of t-SNE embeddings"

    # Layout adjustments
    fig.update_annotations(
        font=dict(size=20, color="white", family="Courier New, monospace")
    )
    fig.update_layout(
        title=dict(
            text=f"{title}",
            x=0.5,
            y=0.99,
            yanchor="top",
            font=dict(size=20, color="white", family="Courier New, monospace"),
        ),
        template="plotly_dark",
        height=800,
        width=1600,
        showlegend=False,
    )

    fig.write_image(f"figures/{fig_title} (plotly).pdf")
    fig.show()


def plot_swiss_roll_plotly(
    sr_points: np.ndarray,
    sr_color: np.ndarray,
    n_samples: int,
    row=1,
    col=1,
    fig=None,
    title=None,
    width=800,
    height=600,
    marker_size=3,
) -> go.Figure:
    """
    Create a 3D scatter plot of the Swiss Roll dataset using Plotly.

    Parameters
    ----------
    sr_points : np.ndarray
        The Swiss Roll dataset points.
    sr_color : np.ndarray
        The colors corresponding to each point.
    n_samples : int
        The number of samples in the dataset.
    row : int, optional
        The row position in a subplot grid. Default is 1.
    col : int, optional
        The column position in a subplot grid. Default is 1.
    fig : go.Figure, optional
        An existing figure object to add the Swiss Roll plot into.

    Returns
    -------
    go.Figure
        The Plotly figure object with the Swiss Roll 3D plot.
    """

    # If no existing figure is passed, create a new one
    if fig is None:
        fig = make_subplots(rows=row, cols=col, specs=[[{"type": "scene"}] * col] * row)

    # Add the Swiss Roll 3D scatter plot
    fig.add_trace(
        go.Scatter3d(
            x=sr_points[:, 0],
            y=sr_points[:, 1],
            z=sr_points[:, 2],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=sr_color,
                colorscale="Rainbow",  # Choose a colorscale
                opacity=0.8,
            ),
        ),
        row=row,
        col=col,
    )
    # title_font_size = width // 40
    # Update the layout for a dark background
    fig.update_layout(
        title="Swiss Roll in Ambient Space" if title is None else title,
        template="plotly_dark",
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            camera=dict(
                eye=dict(x=-1, y=2, z=0.5),  # Camera perspective
            ),
        ),
        width=width * col,
        height=height * row,
        font=dict(family="Courier New, monospace", size=14),
    )

    fig.update_xaxes(
        title_text="x",
        title_font=dict(size=12),  # Set the font size for the x-axis title
        tickfont=dict(size=10),  # Set the font size for the x-axis tick labels
    )
    fig.update_yaxes(
        title_text="y",
        title_font=dict(size=12),  # Set the font size for the y-axis title
        tickfont=dict(size=10),  # Set the font size for the y-axis tick labels
    )

    # Add a text annotation for the number of samples
    fig.add_annotation(
        text=f"N samples={n_samples}",
        xref="paper",
        yref="paper",
        x=1,
        y=0.05,
        showarrow=False,
        font=dict(color="white"),
    )
    fig.show()
    return fig
