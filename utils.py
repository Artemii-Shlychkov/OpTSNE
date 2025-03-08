import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

    return fig
