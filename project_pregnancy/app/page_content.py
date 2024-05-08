"""Functions to generate the content of the different pages of the app."""

import os
import random

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs

import project_pregnancy.default_config as default_config
import src.datasets.utils as data_utils
import src.setcwd
from src.preprocessing import smoothing
from src.regression import training

src.setcwd.main()

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

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
title_fontsize = "40px"

img_herbrain = html.Img(
    src="assets/herbrain.png",
    style={"width": "100%", "height": "auto"},
)


def hormone_slider(hormone_name, hormones_info):
    """Return a slider for a hormone."""
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


def sidebar():
    """Return the sidebar of the app."""
    return html.Div(
        [
            html.H2("HerBrain", className="display-4"),
            html.Hr(),
            html.P(
                "Explore how the female brain changes during pregnancy",
                className="lead",
            ),
            dbc.Nav(
                [
                    dbc.NavLink("Home", href="/", active="exact"),
                    dbc.NavLink("Explore MRI Data", href="/page-1", active="exact"),
                    dbc.NavLink(
                        "AI: Hormones to Hippocampus Shape",
                        href="/page-2",
                        active="exact",
                    ),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )


def homepage():
    """Return the content of the homepage."""
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

    intro_text_row = dbc.Row(
        [dbc.Col(img_herbrain, md=4), dbc.Col(md=1), dbc.Col(intro_text, md=7)],
        style={"marginLeft": margin_side, "marginRight": margin_side},
    )

    return dbc.Container(
        [
            *banner,
            intro_text_row,
        ]
    )


def coordinate_slider(coordinate_name, mri_coordinates_info):
    """Return a slider for a coordinate (coordinate for mri slice)."""
    return dcc.Slider(
        id=f"{coordinate_name}-slider",
        min=mri_coordinates_info[coordinate_name]["min_value"],
        max=mri_coordinates_info[coordinate_name]["max_value"],
        step=mri_coordinates_info[coordinate_name]["step"],
        value=mri_coordinates_info[coordinate_name]["mean_value"],
        marks={
            mri_coordinates_info[coordinate_name]["min_value"]: {"label": "min"},
            mri_coordinates_info[coordinate_name]["max_value"]: {"label": "max"},
        },
        tooltip={
            "placement": "bottom",
            "always_visible": True,
            "style": {"fontSize": "30px", "fontFamily": text_fontfamily},
        },
    )


def explore_data(mri_coordinates_info):
    """Return the content of the data exploration page."""
    banner = [
        html.Div(style={"height": "20px"}),
        html.P(
            [html.Br(), "Explore Brain MRIs Throughout Pregnancy"],
            style={"fontSize": title_fontsize, "fontFamily": text_fontfamily},
        ),
        html.P(
            [html.Br(), "Use the sliders to adjust the MRI slice position."],
            style={"fontSize": text_fontsize, "fontFamily": text_fontfamily},
        ),
    ]

    sliders_card = dbc.Card(
        [
            dbc.Stack(
                [
                    # dbc.Label(
                    #     f"Gestational Week",
                    #     style={"font-size": text_fontsize, "fontFamily": text_fontfamily},
                    # ),
                    # coordinate_slider("week", 0, 40, 1, 20),
                    dbc.Label(
                        "X Coordinate (Changes Side View)",
                        style={
                            "font-size": text_fontsize,
                            "fontFamily": text_fontfamily,
                        },
                    ),
                    coordinate_slider("x", mri_coordinates_info),
                    dbc.Label(
                        "Y Coordinate (Changes Front View)",
                        style={
                            "font-size": text_fontsize,
                            "fontFamily": text_fontfamily,
                        },
                    ),
                    coordinate_slider("y", mri_coordinates_info),
                    dbc.Label(
                        "Z Coordinate (Changes Top View)",
                        style={
                            "font-size": text_fontsize,
                            "fontFamily": text_fontfamily,
                        },
                    ),
                    coordinate_slider("z", mri_coordinates_info),
                ],
                gap=3,
            )
        ],
        body=True,
    )

    sliders_column = [
        dbc.Row(sliders_card),
    ]

    plots_card = dbc.Card(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            dcc.Graph(id="nii-plot-side"),
                            style={"paddingTop": "0px"},
                            # style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'bottom'}
                        ),
                        sm=4,
                        width=700,
                    ),
                    dbc.Col(
                        html.Div(
                            dcc.Graph(id="nii-plot-front"),
                            style={"paddingTop": "0px"},
                            # style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'bottom'}
                        ),
                        sm=4,
                        width=700,
                    ),
                    dbc.Col(
                        html.Div(
                            dcc.Graph(id="nii-plot-top"),
                            style={"paddingTop": "0px"},
                            # style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'bottom'}
                        ),
                        sm=4,
                        width=700,
                    ),
                ],
                align="center",
                style={
                    "marginLeft": margin_side,
                    "marginRight": margin_side,
                    "marginTop": "50px",
                },
            ),
        ],
        body=True,
    )

    return dbc.Container(
        [
            *banner,
            dbc.Row(
                [
                    dbc.Col(plots_card, sm=12, width=100),
                ],
                align="center",
                style={
                    "marginLeft": margin_side,
                    "marginRight": margin_side,
                    "marginTop": "50px",
                },
            ),
            dbc.Row(
                [
                    # dbc.Col(sm=1, width=100),
                    dbc.Col(sliders_column, sm=7, width=700),
                ],
                align="center",
                style={
                    "marginLeft": margin_side,
                    "marginRight": margin_side,
                    "marginTop": "50px",
                },
            ),
            html.Div(style={"height": "20px"}),
        ],
        fluid=True,
    )


def ai_hormone_prediction(
    hormones_info,
):  # estrogen_slider, progesterone_slider, LH_slider, mesh_plot,
    """Return the content of the AI hormone prediction page."""
    sliders_card = dbc.Card(
        [
            dbc.Stack(
                [
                    dbc.Label(
                        "Estrogen pg/ml",
                        style={"font-size": 30, "fontFamily": text_fontfamily},
                    ),
                    # estrogen_slider,
                    hormone_slider("estrogen", hormones_info),
                    dbc.Label(
                        "Progesterone ng/ml",
                        style={"font-size": 30, "fontFamily": text_fontfamily},
                    ),
                    # progesterone_slider,
                    hormone_slider("progesterone", hormones_info),
                    dbc.Label(
                        "LH ng/ml",
                        style={"font-size": 30, "fontFamily": text_fontfamily},
                    ),
                    # LH_slider,
                    hormone_slider("LH", hormones_info),
                ],
                gap=3,
            )
        ],
        body=True,
    )

    sliders_column = [
        # dbc.Row(
        #     html.P(
        #         [
        #             html.Br(),
        #             "Use the sliders to adjust the hormone levels and observe predicted shape changes in the left hippocampal formation.",
        #         ],
        #         style={"fontSize": text_fontsize, "fontFamily": text_fontfamily},
        #     )
        # ),
        dbc.Row(sliders_card),
    ]

    banner = [
        html.Div(style={"height": "20px"}),
        html.P(
            [html.Br(), "AI: Hormones to Hippocampus Shape"],
            style={"fontSize": title_fontsize, "fontFamily": text_fontfamily},
        ),
        html.P(
            [
                html.Br(),
                "Use the sliders to adjust the hormone levels and observe predicted shape changes in the left hippocampal formation.",
            ],
            style={"fontSize": text_fontsize, "fontFamily": text_fontfamily},
        ),
    ]

    return dbc.Container(
        [
            *banner,
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            dcc.Graph(id="mesh-plot"),
                            style={"paddingTop": "0px"},
                            # style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'bottom'}
                        ),
                        # mesh_plot,
                        sm=4,
                        width=700,
                    ),
                    dbc.Col(sm=3, width=100),
                    dbc.Col(sliders_column, sm=4, width=700),
                ],
                align="center",
                style={
                    "marginLeft": margin_side,
                    "marginRight": margin_side,
                    "marginTop": "50px",
                },
            ),
            html.Div(style={"height": "20px"}),
        ],
        fluid=True,
    )
