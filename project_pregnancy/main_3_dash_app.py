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
import nibabel as nib

import project_pregnancy.app.calculations as calculations
import project_pregnancy.app.page_content as page_content
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
) = data_utils.load_real_data(default_config, return_og_segmentation=False)
# Do not include postpartum values that are too low
hormones_df = hormones_df[hormones_df["EndoStatus"] == "Pregnant"]
mesh_sequence_vertices = mesh_sequence_vertices[
    :9
]  # HACKALART: first 9 meshes are pregnancy

lr, pca, X_mean, y_mean, n_vertices, mesh_neighbors = calculations.train_lr_model(
    hormones_df, mesh_sequence_vertices
)

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

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

mri_coordinates_info = {
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

pre_calculated_mri_slices = calculations.pre_calculate_mri_slices(mri_coordinates_info)

sidebar = page_content.sidebar()

home_page = page_content.homepage()
explore_data_page = page_content.explore_data()
ai_hormone_prediction_page = page_content.ai_hormone_prediction(hormones_info)

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}
content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    """Render the page content based on the URL."""
    if pathname == "/":
        return home_page
    elif pathname == "/page-1":
        return explore_data_page
    elif pathname == "/page-2":
        return ai_hormone_prediction_page
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


@app.callback(
    Output("mesh-plot", "figure"),
    Input("estrogen-slider", "drag_value"),
    Input("progesterone-slider", "drag_value"),
    Input("LH-slider", "drag_value"),
    State("mesh-plot", "figure"),
    State("mesh-plot", "relayoutData"),
)
def update_mesh(estrogen, progesterone, LH, current_figure, relayoutData):
    """Update the mesh plot based on the hormone levels."""
    return calculations.predict_mesh(
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
        current_figure=current_figure,
        relayoutData=relayoutData,
    )


@app.callback(
    [
        Output("nii-plot-side", "figure"),
        Output("nii-plot-front", "figure"),
        Output("nii-plot-top", "figure"),
    ],
    # Input("gestation-week-slider", "drag_value"),
    Input("x-slider", "drag_value"),
    Input("y-slider", "drag_value"),
    Input("z-slider", "drag_value"),
)
def update_nii_plot(x, y, z):  # week,
    """Update the nii plot based on the week and the x, y, z coordinates."""
    side, front, top = calculations.return_nii_plot(x, y, z)
    return side, front, top


if __name__ == "__main__":
    app.run_server(
        debug=True, use_reloader=True, host="0.0.0.0", port="8050"
    )  # Turn off reloader if inside Jupyter
