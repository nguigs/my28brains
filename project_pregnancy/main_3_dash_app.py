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

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs

import project_pregnancy.default_config as default_config
import src.datasets.utils as data_utils
import src.setcwd
from src.regression import training

src.setcwd.main()

# Multiple Linear Regression
# Note:
# -true intercept is the first mesh after reparametrization
# -true coef is the difference between the first two meshes after reparametrization
# The reparameterization does not perform regression, thus they are not (neither true, not regression estimated)
# intercept and coef

(
    space,
    mesh_sequence_vertices,
    vertex_colors,
    hormones_df,
) = data_utils.load_real_data(default_config)

n_vertices = len(mesh_sequence_vertices[0])
n_meshes_in_sequence = len(mesh_sequence_vertices)
faces = gs.array(space.faces).numpy()

# TODO: instead, save these values in main_2, and then load them here. or, figure out how to predict the mesh using just the intercept and coef learned here, and then load them.

progesterone_levels = gs.array(hormones_df["prog"].values)
estrogen_levels = gs.array(hormones_df["estro"].values)
lh_levels = gs.array(hormones_df["lh"].values)
# gest_week = gs.array(all_hormone_levels["gestWeek"].values)

progesterone_average = gs.mean(progesterone_levels)
estrogen_average = gs.mean(estrogen_levels)
lh_average = gs.mean(lh_levels)
# gest_week_average = gs.mean(gest_week)

y = gs.array(mesh_sequence_vertices)
X_multiple = gs.vstack(
    (
        progesterone_levels,
        estrogen_levels,
        lh_levels,
        # gest_week,
    )
).T  # NOTE: copilot thinks this should be transposed.

(
    multiple_intercept_hat,
    multiple_coef_hat,
    mr,
    p_values,
) = training.fit_linear_regression(y, X_multiple, return_p=True)

# NOTE (Nina): this is not really n_train
# since we've just trained on the whole dataset
n_train = int(default_config.train_test_split * n_meshes_in_sequence)

X_indices = np.arange(n_meshes_in_sequence)
# Shuffle the array to get random values
random.shuffle(X_indices)
train_indices = X_indices[:n_train]
train_indices = np.sort(train_indices)
test_indices = X_indices[n_train:]
test_indices = np.sort(test_indices)
mr_score_array = training.compute_R2(y, X_multiple, test_indices, train_indices)

# hormone p values
progesterone_p_value = p_values[0]
estrogen_p_value = p_values[1]
lh_p_value = p_values[2]

# Parameters for sliders

hormones_info = {
    "progesterone": {"min_value": 1, "max_value": 103, "step": 10},
    "estrogen": {"min_value": 3, "max_value": 10200, "step": 100},
    "LH": {"min_value": 1, "max_value": 8, "step": 1},
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
    y_pred_for_mr = mr.predict(X_multiple)
    y_pred_for_mr = y_pred_for_mr.reshape([n_vertices, 3])
    # y_pred_for_mr = gaussian_smoothing(y_pred_for_mr, sigma=0.7)

    # Plot Mesh
    faces = gs.array(space.faces).numpy()
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
                x=y_pred_for_mr[:, 0],
                y=y_pred_for_mr[:, 1],
                z=y_pred_for_mr[:, 2],
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
