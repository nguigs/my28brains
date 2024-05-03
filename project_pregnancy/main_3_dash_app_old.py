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
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs

import project_pregnancy.default_config as default_config
import src.datasets.utils as data_utils
import src.setcwd
from src.preprocessing import smoothing
from src.regression import training

src.setcwd.main()

(
    space,
    mesh_sequence_vertices,
    vertex_colors,
    hormones_df,
) = data_utils.load_real_data(default_config, return_raw_data=False)
# Do not include postpartum values that are too low
hormones_df = hormones_df[hormones_df["EndoStatus"] == "Pregnant"]
mesh_sequence_vertices = mesh_sequence_vertices[
    :9
]  # HACKALART: first 9 meshes are pregnancy
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

# # NOTE (Nina): this is not really n_train
# # since we've just trained on the whole dataset
# # TODO: FIX THIS BLOCK
# n_train = int(default_config.train_test_split * n_meshes)
# X_indices = np.arange(n_meshes)
# # Shuffle the array to get random values
# random.shuffle(X_indices)
# train_indices = X_indices[:n_train]
# train_indices = np.sort(train_indices)
# test_indices = X_indices[n_train:]
# test_indices = np.sort(test_indices)
# mr_score_array = training.compute_R2(y[:9], X, test_indices, train_indices)
p_values = [
    0.0,
    0.0,
    0.0,
]  # placeholder, else: training.calculate_p_values(X_multiple, y_pca, lr)
estrogen_p_value = p_values[0]
progesterone_p_value = p_values[1]
lh_p_value = p_values[2]

hormones_info = {
    "estrogen": {
        "min_value": 4100,
        "max_value": 12400,
        "mean_value": X_mean[0],
        "step": 500,
    },
    "progesterone": {
        "min_value": 54,
        "max_value": 103,
        "mean_value": X_mean[1],
        "step": 3,
    },
    "LH": {"min_value": 0.59, "max_value": 1.45, "mean_value": X_mean[2], "step": 0.05},
}

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
banner = [
    html.Div(style={"height": "20px"}),
    html.Img(
        src="assets/herbrain_logo_text.png",
        style={
            "width": "80%",
            "height": "auto",
            "marginLeft": "10px",
            "marginRight": "10px",
        },
    ),
    html.Hr(),
]

margin_side = "20px"
text_fontsize = "24px"
text_fontfamily = "Avenir"
intro_text = html.P(
    [
        html.Br(),
        "The hippocampus and the structures around it are particularly sensitives to hormones.",
        html.Br(),
        html.Br(),
        "In pregnancy, sex hormones are believed to drive the decline in hippocampal volume that occurs during gestation.",
        html.Br(),
        html.Br(),
        "This application predicts the shape changes occuring in the hippocampus during pregnancy based on hormone levels.",
    ],
    style={"fontSize": text_fontsize, "fontFamily": text_fontfamily},
)
img_herbrain = html.Img(
    src="assets/herbrain.png",
    style={"width": "100%", "height": "auto"},
)
intro_text_row = dbc.Row(
    [dbc.Col(img_herbrain, md=4), dbc.Col(md=1), dbc.Col(intro_text, md=7)],
    style={"marginLeft": margin_side, "marginRight": margin_side},
)


def hormone_slider(hormone_name):
    """Create a slider for a hormone."""
    return dcc.Slider(
        id=f"{hormone_name}-slider",
        min=hormones_info[hormone_name]["min_value"],
        max=hormones_info[hormone_name]["max_value"],
        step=hormones_info[hormone_name]["step"],
        value=hormones_info[hormone_name]["mean_value"],
        marks={
            hormones_info[hormone_name]["min_value"]: {"label": "min"},
            hormones_info[hormone_name]["max_value"]: {"label": "max"},
        },
        tooltip={
            "placement": "bottom",
            "always_visible": True,
            "style": {"fontSize": "30px", "fontFamily": text_fontfamily},
        },
    )


sliders_card = dbc.Card(
    [
        dbc.Stack(
            [
                dbc.Label(
                    "Estrogen pg/ml",
                    style={"font-size": 30, "fontFamily": text_fontfamily},
                ),
                hormone_slider("estrogen"),
                dbc.Label(
                    "Progesterone ng/ml",
                    style={"font-size": 30, "fontFamily": text_fontfamily},
                ),
                hormone_slider("progesterone"),
                dbc.Label(
                    "LH ng/ml", style={"font-size": 30, "fontFamily": text_fontfamily}
                ),
                hormone_slider("LH"),
            ],
            gap=3,
        )
    ],
    body=True,
)

sliders_column = [
    dbc.Row(
        html.P(
            [
                html.Br(),
                "Use the sliders to adjust the hormone levels and observe predicted shape changes in the left hippocampal formation.",
            ],
            style={"fontSize": text_fontsize, "fontFamily": text_fontfamily},
        )
    ),
    dbc.Row(sliders_card),
]

app.layout = dbc.Container(
    [
        *banner,
        intro_text_row,
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        dcc.Graph(id="mesh-plot"),
                        style={"paddingTop": "-20px"},  # Adjust the padding top style
                    ),
                    md=4,
                ),
                dbc.Col(md=1),
                dbc.Col(sliders_column, md=7),
            ],
            align="center",
            style={
                "marginLeft": margin_side,
                "marginRight": margin_side,
                "marginTop": "-50px",
            },
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        dcc.Graph(id="real-data-plot"),
                        style={"paddingTop": "-20px"},  # Adjust the padding top style
                    ),
                    md=4,
                ),
                dbc.Col(md=1),
            ],
            align="center",
            style={
                "marginLeft": margin_side,
                "marginRight": margin_side,
                "marginTop": "-50px",
            },
        ),
        html.Div(style={"height": "20px"}),
    ],
    fluid=True,
)


@callback(
    Output("mesh-plot", "figure"),
    Input("estrogen-slider", "drag_value"),
    Input("progesterone-slider", "drag_value"),
    Input("LH-slider", "drag_value"),
    State("mesh-plot", "figure"),
    State("mesh-plot", "relayoutData"),
)
def update_mesh(estrogen, progesterone, LH, current_figure, relayoutData):
    """Update the mesh plot based on the hormone levels."""
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


@callback(
    Output("real-data-plot", "figure"),
    Input("estrogen-slider", "drag_value"),
    Input("progesterone-slider", "drag_value"),
    Input("LH-slider", "drag_value"),
    State("real-data-plot", "figure"),
    State("real-data-plot", "relayoutData"),
)
def update_real_data(estrogen, progesterone, LH, current_figure, relayoutData):
    """Update the real data plot based on the hormone levels.

    gives the real data mesh whose hormone levels are the closest to the sliders.
    """
    # Find the closest hormone levels
    real_meshes = mesh_sequence_vertices
    real_hormones = hormones_df

    slider_hormones = np.array([estrogen, progesterone, LH])
    real_hormones = real_hormones[["estro", "prog", "lh"]].values
    distances = np.linalg.norm(real_hormones - slider_hormones, axis=1)
    closest_index = np.argmin(distances)
    # closest_hormones = real_hormones[closest_index]
    closest_mesh = real_meshes[closest_index]

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
                x=closest_mesh[:, 0],
                y=closest_mesh[:, 1],
                z=closest_mesh[:, 2],
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
        # xaxis_title=f"estrogen: {closest_hormones[0]}, progesterone: {closest_hormones[1]}, LH: {closest_hormones[2]}",
    )

    # fig.add_trace(go.Scatter(
    #     x=[0],
    #     y=[0],
    #     mode="text",
    #     name="Text",
    #     text=[f"estrogen: {closest_hormones[0]}, progesterone: {closest_hormones[1]}, LH: {closest_hormones[2]}"],
    #     textposition="bottom center"
    # ))

    # fig.update_layout(
    #     legend={"estrogen": closest_hormones[0], "progesterone": closest_hormones[1], "LH": closest_hormones[2]}
    # )

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
    app.run_server(
        debug=True, use_reloader=True, host="0.0.0.0", port="8050"
    )  # Turn off reloader if inside Jupyter
