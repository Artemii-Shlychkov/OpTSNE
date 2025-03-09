import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import sklearn.datasets
from plotly.subplots import make_subplots
from tensorflow.keras.datasets import mnist

from tsne_api import TSNEResult, TSNEResultsWithKNN


@dataclass
class SwissRoll:
    datapoints: np.ndarray
    labels: np.ndarray
    n_samples: int

    @classmethod
    def generate(cls, n_samples: int, noise: float) -> "SwissRoll":
        datapoints, labels = sklearn.datasets.make_swiss_roll(
            n_samples=n_samples, noise=noise
        )
        return SwissRoll(datapoints=datapoints, labels=labels, n_samples=n_samples)

    def plot(
        self, width: int = 600, height: int = 600, title: str | None = None
    ) -> None:
        plot_swiss_roll(
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
    def generate(cls, n_training_samples: int) -> "MNIST":
        (data_train, labels_train), (data_test, labels_test) = mnist.load_data()
        data_train = data_train[:n_training_samples]
        labels_train = labels_train[:n_training_samples]
        return MNIST(data_train, labels_train, data_test, labels_test)

    def reshape(self) -> None:
        self.data_train = self.data_train.reshape(self.data_train.shape[0], -1)
        self.data_test = self.data_test.reshape(self.data_test.shape[0], -1)


### PLOTTING FUNCTIONS ###


def plot_tsne_result(
    data: TSNEResult | TSNEResultsWithKNN,
    labels: np.ndarray,
    additional_title: str = "",
    marker_size: int = 3,
    *,
    black_template: bool = False,
) -> None:
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
            marker={
                "color": labels,
                "size": marker_size,
                "showscale": False,
                "colorscale": "Jet",
            },
            name="t-SNE Embedding",
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)

    # KL Divergence Plot
    min_kl = data.im_KLs[-1]
    im_kls_filtered = [kl for kl in data.im_KLs if kl != 0]
    fig.add_trace(
        go.Scatter(
            x=list(range(len(im_kls_filtered))),
            y=im_kls_filtered,
            mode="markers+lines",
            marker={"size": 2, "color": "blue"},
            line={"color": "blue"},
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
            line={"color": "red"},
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
            line={"color": "green"},
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
            tickvals=[0, len(data.im_KLs) // 2, len(data.im_KLs)],
            ticktext=["0", f"{len(data.im_KLs) // 2}", f"{len(data.im_KLs)}"],
            range=[0, len(data.im_KLs)],
            row=2,
            col=i,
        )

    # Titles and Layout
    fig.update_annotations(
        font={"size": 16, "color": text_color, "family": "Courier New, monospace"}
    )
    axis_label_style = {
        "size": 16,
        "color": text_color,
        "family": "Courier New, monospace",
    }
    fig.update_xaxes(
        title={"text": "Iterations", "font": axis_label_style}, row=2, col=2
    )
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
        title={
            "text": (
                f"{fig_title}{title_param_2}<br>{title_param_3}<br>{additional_title}"
            ),
            "x": 0.5,
            "y": 0.95,
            "yanchor": "top",
            "font": {
                "size": 16,
                "color": text_color,
                "family": "Courier New, monospace",
            },
        },
        template=template,
        height=800,
        width=1000,
        showlegend=False,
    )

    # Sanitize the fig_title to remove invalid characters
    sanitized_fig_title = re.sub(r'[<>:"/\\|?*]', "_", fig_title)

    fig_path = Path(f"figures/{sanitized_fig_title} (plotly).pdf")
    fig_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    with fig_path.open("wb") as f:
        f.write(fig.to_image(format="pdf"))

    fig.show()


def plot_side_by_side(
    data1: TSNEResult | TSNEResultsWithKNN,
    data2: TSNEResult | TSNEResultsWithKNN,
    labels: np.ndarray,
    additional_title_1: str = "",
    additional_title_2: str = "",
    marker_size: int = 3,
    title: str = "",
) -> None:
    # Extract titles for the embeddings
    def get_titles(
        data: TSNEResult | TSNEResultsWithKNN,
        knn_recall: float,
        additional_title: str = "",
    ) -> str:
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

        return f"t-SNE embedding with {mode} α parameter<br>{params}<br>{summary}<br>{additional_title}"

    knn_recall_data1 = getattr(data1, "knn_recall", "N/A")
    knn_recall_data2 = getattr(data2, "knn_recall", "N/A")

    title1 = get_titles(data1, knn_recall_data1, additional_title_1)
    title2 = get_titles(data2, knn_recall_data2, additional_title_2)

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
    x_ticks = [0, len(data1.im_KLs) // 2, len(data1.im_KLs)]

    def add_embedding(
        data: TSNEResult | TSNEResultsWithKNN, col: int, row: int, labels: np.ndarray
    ) -> None:
        fig.add_trace(
            go.Scatter(
                x=data.embedding[:, 0],
                y=data.embedding[:, 1],
                mode="markers",
                marker={
                    "color": labels,
                    "size": marker_size,
                    "showscale": False,
                    "colorscale": "Rainbow",
                },
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
        col: int,
        row: int,
        metric: list[float],
        name: str,
        line_color: str,
        hline: float | None = None,
        hline_color: str | None = None,
    ) -> None:
        # exclude 0 values from metric
        metric = [m for m in metric if m != 0]

        fig.add_trace(
            go.Scatter(
                x=list(range(len(metric))),
                y=metric,
                mode="markers+lines",
                marker={"size": 2, "color": line_color},
                line={"color": line_color},
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
            range=[0, len(metric)],
            title={
                "text": "Iterations",
                "font": {"size": 24, "family": "Courier New, monospace"},
            }
            if col in (2, 5)
            else None,
            row=row,
            col=col,
        )

        fig.update_yaxes(row=row, col=col, tickvals=[hline], ticktext=[f"{hline:.2f}"])

    # Add first embedding and metrics
    add_embedding(data1, 1, 1, labels)
    add_metric_plot(
        data1, 1, 2, data1.im_KLs, "min 𝓛", "blue", data1.im_KLs[-1], "orange"
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
        data2, 4, 2, data2.im_KLs, "min 𝓛", "blue", data2.im_KLs[-1], "orange"
    )
    add_metric_plot(
        data2,
        5,
        2,
        data2.im_alphas,
        "Converged α",
        "red",
        data2.im_alphas[-1],
        "green",
    )
    add_metric_plot(
        data2, 6, 2, data2.im_alpha_grads, "Zero Gradient", "green", 0, "red"
    )

    # Generate title if embeddings are the same
    if data1.dataset_name == data2.dataset_name and data1.n_samples == data2.n_samples:
        fig_title = f"Comparison of t-SNE embeddings for {data1.dataset_name} with {data1.n_samples} samples"
    else:
        fig_title = "Comparison of t-SNE embeddings"

    # Layout adjustments
    fig.update_annotations(
        font={"size": 20, "color": "white", "family": "Courier New, monospace"}
    )
    fig.update_layout(
        title={
            "text": f"{title}",
            "x": 0.5,
            "y": 0.99,
            "yanchor": "top",
            "font": {"size": 20, "color": "white", "family": "Courier New, monospace"},
        },
        template="plotly_dark",
        height=800,
        width=1600,
        showlegend=False,
    )

    sanitized_fig_title = re.sub(r'[<>:"/\\|?*]', "_", fig_title)
    fig_path = Path(f"figures/{sanitized_fig_title} (plotly).pdf")
    fig_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    with fig_path.open("wb") as f:
        f.write(fig.to_image(format="pdf"))

    fig.show()


def plot_swiss_roll(
    sr_points: np.ndarray,
    sr_color: np.ndarray,
    n_samples: int,
    row: int = 1,
    col: int = 1,
    fig: go.Figure | None = None,
    title: str | None = None,
    width: int = 600,
    height: int = 600,
    marker_size: int = 3,
) -> go.Figure:
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
            marker={
                "size": marker_size,
                "color": sr_color,
                "colorscale": "Rainbow",  # Choose a colorscale
                "opacity": 0.8,
            },
        ),
        row=row,
        col=col,
    )

    # Update the layout for a dark background
    fig.update_layout(
        title="Swiss Roll in Ambient Space" if title is None else title,
        template="plotly_dark",
        scene={
            "xaxis": {"title": "X"},
            "yaxis": {"title": "Y"},
            "zaxis": {"title": "Z"},
            "camera": {
                "eye": {"x": -1, "y": 2, "z": 0.5},  # Camera perspective
            },
        },
        width=width * col,
        height=height * row,
        font={"family": "Courier New, monospace", "size": 14},
    )

    fig.update_xaxes(
        title_text="x",
        title_font={"size": 12},  # Set the font size for the x-axis title
        tickfont={"size": 10},  # Set the font size for the x-axis tick labels
    )
    fig.update_yaxes(
        title_text="y",
        title_font={"size": 12},  # Set the font size for the y-axis title
        tickfont={"size": 10},  # Set the font size for the y-axis tick labels
    )

    # Add a text annotation for the number of samples
    fig.add_annotation(
        text=f"N samples={n_samples}",
        xref="paper",
        yref="paper",
        x=1,
        y=0.05,
        showarrow=False,
        font={"color": "white"},
    )
    fig.show()
    return fig
