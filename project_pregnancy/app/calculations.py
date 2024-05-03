"""Functions perform calculations necessary for the dash app."""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs
import nibabel as nib

import project_pregnancy.default_config as default_config
import src.datasets.utils as data_utils
import src.setcwd
import src.viz as viz
from src.preprocessing import smoothing
from src.regression import training

src.setcwd.main()


def train_lr_model(hormones_df, mesh_sequence_vertices, p_values=False):
    """Train a linear regression model on the data."""
    mean_mesh = mesh_sequence_vertices.mean(axis=0)

    # Compute neighbors once and for all from the mean mesh
    k_neighbors = 10
    mesh_neighbors = smoothing.compute_neighbors(mean_mesh, k=k_neighbors)

    n_meshes, n_vertices, _ = mesh_sequence_vertices.shape

    X = hormones_df[["estro", "prog", "lh"]].values
    _, n_hormones = X.shape
    X_mean = X.mean(axis=0)

    y = mesh_sequence_vertices.reshape(n_meshes, -1)
    y_mean = y.mean(axis=0)
    y = y - y_mean

    # Define the number of principal components
    n_components = 4  # Adjust based on variance explanation: see notebook 02
    pca = PCA(n_components=n_components)
    y_pca = pca.fit_transform(y)
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"Cumul. variance explained w/ {n_components} components: {explained_var}")

    lr = LinearRegression()
    lr.fit(X, y_pca)
    y_pca_pred = lr.predict(X)
    r2 = r2_score(y_pca, y_pca_pred)
    adjusted_r2 = 1 - (1 - r2) * (n_meshes - 1) / (n_meshes - n_hormones - 1)
    print(f"Adjusted R2 score (adjusted for several inputs): {adjusted_r2:.2f}")

    if p_values:
        p_values = [
            0.0,
            0.0,
            0.0,
        ]  # placeholder, else: training.calculate_p_values(X_multiple, y_pca, lr)
        estrogen_p_value = p_values[0]
        progesterone_p_value = p_values[1]
        lh_p_value = p_values[2]

        return (
            lr,
            pca,
            X_mean,
            y_mean,
            n_vertices,
            mesh_neighbors,
            adjusted_r2,
            estrogen_p_value,
            progesterone_p_value,
            lh_p_value,
        )

    return lr, pca, X_mean, y_mean, n_vertices, mesh_neighbors


def predict_mesh(
    estrogen,
    progesterone,
    LH,
    lr,
    pca,
    y_mean,
    n_vertices,
    mesh_neighbors,
    space,
    vertex_colors,
    current_figure=None,
    relayoutData=None,
):
    """Predict the mesh based on the hormone values."""
    # Predict Mesh
    X_multiple = gs.array([[estrogen, progesterone, LH]])
    y_pca_pred = lr.predict(X_multiple)

    y_pred = pca.inverse_transform(y_pca_pred) + y_mean.numpy()
    mesh_pred = y_pred.reshape(n_vertices, 3)
    mesh_pred = smoothing.median_smooth(mesh_pred, mesh_neighbors)

    # Plot Mesh
    if current_figure and "layout" in current_figure:
        layout = current_figure["layout"]
    else:
        layout = go.Layout(
            margin=go.layout.Margin(
                l=0,
                r=0,
                b=0,
                t=0,
            ),
            width=700,
            height=700,
            scene=dict(
                aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"
            ),
        )

    faces = gs.array(space.faces).numpy()
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=mesh_pred[:, 0],
                y=mesh_pred[:, 1],
                z=mesh_pred[:, 2],
                colorbar_title="z",
                vertexcolor=vertex_colors,
                # i, j and k give the vertices of triangles
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                name="y",
            )
        ],
        layout=layout,
    )

    if relayoutData and ("scene.camera" in relayoutData):
        scene_camera = relayoutData["scene.camera"]
    else:
        scene_camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=0, z=2.5),
        )
    fig.update_layout(scene_camera=scene_camera)
    return fig


def plot_slice_as_plotly(
    one_slice, cmap="gray", title="Slice Visualization", x_label="X", y_label="Y"
):
    """Display an image slice as a Plotly figure."""
    # Create heatmap trace for the current slice
    heatmap_trace = go.Heatmap(z=one_slice.T, colorscale=cmap)  # , zmin=0, zmax=1)

    # Create a Plotly figure with the heatmap trace
    fig = go.Figure(data=heatmap_trace)

    # Update layout to adjust appearance
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)

    return fig


def show_slices(slices, cmap="gray"):
    """Display row of image slices.

    Parameters
    ----------
    slices : list
        List of integers that represent the indexes of the slices to plot.
    """
    fig, axes = plt.subplots(1, len(slices))
    for i_slice, one_slice in enumerate(slices):
        im = axes[i_slice].imshow(one_slice.T, cmap=cmap, origin="lower")
    return fig, axes, im


def show_slice(
    one_slice, cmap="gray", title="Slice Visualization", x_label="X", y_label="Y"
):
    """Display an image slice.

    Parameters
    ----------
    one_slice : ndarray
        2D array representing the image slice.
    cmap : str, optional
        Colormap to use for the visualization.
    """
    # Create a new figure
    fig, ax = plt.subplots()

    # Display the slice
    im = ax.imshow(one_slice.T, cmap=cmap, origin="lower")

    # Customize the plot
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.colorbar(im, ax=ax)

    return fig


def return_nii_plot(x, y, z):  # week,
    """Return the nii plot based on the week and the x, y, z coordinates."""
    PREGNANCY_DIR = "/home/data/pregnancy"
    img_path = os.path.join(PREGNANCY_DIR, "BrainNormalizedToTemplate.nii.gz")
    img = nib.load(img_path)
    img_data = img.get_fdata()

    slice_0 = img_data[x, :, :]  # was 206
    slice_1 = img_data[:, y, :]  # was 130
    slice_2 = img_data[:, :, z]  # was 160

    side_fig = plot_slice_as_plotly(
        slice_0, cmap="gray", title="Side View", x_label="Y", y_label="Z"
    )
    front_fig = plot_slice_as_plotly(
        slice_1, cmap="gray", title="Front View", x_label="X", y_label="Z"
    )
    top_fig = plot_slice_as_plotly(
        slice_2, cmap="gray", title="Top View", x_label="X", y_label="Y"
    )

    return side_fig, front_fig, top_fig


def pre_calculate_mri_slices(mri_coordinates_info):
    """Pre-calculate the slices of the MRI image."""
    PREGNANCY_DIR = "/home/data/pregnancy"
    img_path = os.path.join(PREGNANCY_DIR, "BrainNormalizedToTemplate.nii.gz")
    img = nib.load(img_path)
    img_data = img.get_fdata()

    slice_0 = img_data[206, :, :]
    slice_1 = img_data[:, 130, :]
    slice_2 = img_data[:, :, 160]

    return slice_0, slice_1, slice_2
