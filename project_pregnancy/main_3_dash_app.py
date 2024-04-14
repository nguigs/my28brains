"""Creates a Dash app where hormone sliders predict hippocampal shape.

Run the app with the following command:
python main_3_dash_app.py

Notes on Dash:
- Dash is a Python framework for building web applications.
- html.H1(children= ...) is an example of a title. You can change H1 to H2, H3, etc.
    to change the size of the title.
"""

import os
import random

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go  # or plotly.express as px
from dash import Dash, Input, Output, callback, dcc, html, State
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs

import project_pregnancy.default_config as default_config
import src.datasets.utils as data_utils
import src.setcwd
from src.regression import training

src.setcwd.main()


# Multiple Linear Regression

(
    space,
    mesh_sequence_vertices,
    vertex_colors,
    hormones_df,
) = data_utils.load_real_data(default_config)
faces = gs.array(space.faces).numpy()

n_vertices = len(mesh_sequence_vertices[0])
n_meshes_in_sequence = len(mesh_sequence_vertices)

# TODO: instead, save these values in main_2, and then load them here. 
# or, figure out how to predict the mesh using just the intercept and coef learned here, and then load them.

# Extract only until birth, do not include postpartum values that are too low
birth_id = 9
progesterone_levels = gs.array(hormones_df["prog"].values)[:birth_id]
estrogen_levels = gs.array(hormones_df["estro"].values)[:birth_id]
lh_levels = gs.array(hormones_df["lh"].values)[:birth_id]

progesterone_average = gs.mean(progesterone_levels)
estrogen_average = gs.mean(estrogen_levels)
lh_average = gs.mean(lh_levels)

y = mesh_sequence_vertices

# Define the number of principal components
n_components = 4  # Adjust based on variance explanation
pca = PCA(n_components=n_components)
y_reshaped = y.reshape(n_meshes_in_sequence, -1)
mean_mesh = y_reshaped.mean(axis=0)
y_reshaped = y_reshaped - mean_mesh

y_pca = pca.fit_transform(y_reshaped)
explained_var = np.sum(pca.explained_variance_ratio_)
print(f"The cumulated variance explained with {n_components} components is: {explained_var}")

X_multiple = gs.vstack(
    (
        progesterone_levels,
        estrogen_levels,
        lh_levels,
        # gest_week,
    )
).T  # NOTE: copilot thinks this should be transposed.
lr = LinearRegression()
y = y_pca

# only until 9 because it's post partum after: values are super low.
lr.fit(X_multiple[:9], y[:9])
p_values = [0., 0., 0.]  # placeholder, else: training.calculate_p_values(X_multiple, y_pca, lr)
intercept_hat = lr.intercept_
coef_hat = lr.coef_

# NOTE (Nina): this is not really n_train
# since we've just trained on the whole dataset
n_meshes_in_sequence = len(y[:9])
n_train = int(default_config.train_test_split * n_meshes_in_sequence)

X_indices = np.arange(n_meshes_in_sequence)
# Shuffle the array to get random values
random.shuffle(X_indices)
train_indices = X_indices[:n_train]
train_indices = np.sort(train_indices)
test_indices = X_indices[n_train:]
test_indices = np.sort(test_indices)
mr_score_array = training.compute_R2(y[:9], X_multiple, test_indices, train_indices)

progesterone_p_value = p_values[0]
estrogen_p_value = p_values[1]
lh_p_value = p_values[2]

# Parameters for sliders

hormones_info = {
    "progesterone": {"min_value": 54, "max_value": 103, "step": 5},
    "estrogen": {"min_value": 4100, "max_value": 10200, "step": 100},
    "LH": {"min_value": 0.59, "max_value": 1.45, "step": 0.1},
    # "gest_week": {"min_value": -3, "max_value": 162, "step": 10},
}

app = Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP]
)  # , external_stylesheets=external_stylesheets)

sliders = dbc.Card(
    [
        dbc.Stack(
            [
                # html.H6(f"Progesterone ng/ml, p-value: {progesterone_p_value}"),
                dbc.Label(
                    f"Progesterone ng/ml, % significant p-values: {progesterone_p_value:.2f}",
                    style={"font-size": 30},
                ),
                dcc.Slider(
                    id="progesterone-slider",
                    min=hormones_info["progesterone"]["min_value"],
                    max=hormones_info["progesterone"]["max_value"],
                    step=hormones_info["progesterone"]["step"],
                    value=progesterone_average,
                    marks={
                        hormones_info["progesterone"]["min_value"]: {"label": "min"},
                        hormones_info["progesterone"]["max_value"]: {"label": "max"},
                    },
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                        "style": {"fontSize": "30px"},
                    },
                ),
                # html.H6(f"Estrogen pg/ml, p-value: {estrogen_p_value}"),
                dbc.Label(
                    f"Estrogen pg/ml, % significant p-values: {estrogen_p_value:.2f}",
                    style={"font-size": 30},
                ),
                dcc.Slider(
                    id="estrogen-slider",
                    min=hormones_info["estrogen"]["min_value"],
                    max=hormones_info["estrogen"]["max_value"],
                    step=hormones_info["estrogen"]["step"],
                    value=estrogen_average,
                    marks={
                        hormones_info["estrogen"]["min_value"]: {"label": "min"},
                        hormones_info["estrogen"]["max_value"]: {"label": "max"},
                    },
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                        "style": {"fontSize": "30px"},
                    },
                ),
                # html.H6(f"LH ng/ml, p-value: {lh_p_value}"),
                dbc.Label(
                    f"LH ng/ml, % significant p-values: {lh_p_value:.2f}",
                    style={"font-size": 30},
                ),
                dcc.Slider(
                    id="LH-slider",
                    min=hormones_info["LH"]["min_value"],
                    max=hormones_info["LH"]["max_value"],
                    step=hormones_info["LH"]["step"],
                    value=lh_average,
                    marks={
                        hormones_info["LH"]["min_value"]: {"label": "min"},
                        hormones_info["LH"]["max_value"]: {"label": "max"},
                    },
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                        "style": {"fontSize": "30px"},
                    },
                ),
                # html.H6("Gestation Week"),
                # dcc.Slider(
                #     id="gest_week-slider",
                #     min=hormones_info["gest_week"]["min_value"],
                #     max=hormones_info["gest_week"]["max_value"],
                #     step=hormones_info["gest_week"]["step"],
                #     value=gest_week_average,
                #     marks={
                #         str(i): str(i)
                #         for i in range(
                #             hormones_info["gest_week"]["min_value"],
                #             hormones_info["gest_week"]["max_value"],
                #             hormones_info["gest_week"]["step"],
                #         )
                #     },
                # ),
            ],
            gap=3,
        ),
    ],
    body=True,
)

app.layout = dbc.Container(
    [
        html.H1("Brain Shape Prediction with Hormones, Pregnancy"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(sliders, md=6),
                dbc.Col(dcc.Graph(id="mesh-plot"), md=6),
            ],
            align="center",
        ),
    ],
    fluid=True,
)


@callback(
    Output("mesh-plot", "figure"),
    Input("progesterone-slider", "drag_value"),
    Input("estrogen-slider", "drag_value"),
    Input("LH-slider", "drag_value"),
    State("mesh-plot", "figure"),
    State("mesh-plot", "relayoutData"),
)
def update_mesh(progesterone, estrogen, LH, current_figure, relayoutData):
    """Update the mesh plot based on the hormone levels."""
    # Predict Mesh
    X_multiple = gs.array([[progesterone, estrogen, LH]])
    y_pred = lr.predict(X_multiple)

    y_pred = pca.inverse_transform(y_pred) + mean_mesh.numpy()
    y_pred = y_pred.reshape(n_vertices, 3)

    # y_pred_for_mr = gaussian_smoothing(y_pred_for_mr, sigma=0.7)

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
            width=1000,
            height=1000,
            scene=dict(
                aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"
            ),
        )

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=y_pred[:, 0],
                y=y_pred[:, 1],
                z=y_pred[:, 2],
                colorbar_title="z",
                vertexcolor=vertex_colors,
                # i, j and k give the vertices of triangles
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                name="y",
                # showscale=True,
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


if __name__ == "__main__":
    # app.run(debug=True)
    app.run_server(
        debug=True, use_reloader=False
    )  # Turn off reloader if inside Jupyter
