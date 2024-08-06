# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 19:35:58 2024

@author: edgar
"""

from dash import Dash, dcc, html, Input, Output, callback
import numpy as np
import pandas as pd
import plotly.express as px
import json

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
# make a sample data frame with 6 columns
np.random.seed(234)
df = pd.DataFrame(zip(np.random.rand(30),np.random.rand(30),[str(a) for a in range(30)] ,30*[False]), columns=("x","y","labels","selected"))

fig = px.scatter(df, x="x", y="y")

fig.update_layout(clickmode='event+select')

fig.update_traces(marker_size=20)

app.layout = html.Div([
    dcc.Graph(
        id='basic-interactions',
        figure=fig
    ),
  dcc.Store(id='intermediate-value'),
    html.Div(className='row', children=[
        html.Div([
            dcc.Markdown("""
                **Hover Data**

                Mouse over values in the graph.
            """),
            html.Pre(id='hover-data', style=styles['pre']),
        ], className='three columns'),

        html.Div([
            dcc.Markdown("""
                **Click Data**

                Click on points in the graph.
            """),
            html.Pre(id='click-data', style=styles['pre']),
        ], className='three columns'),

        html.Div([
            dcc.Markdown("""
                **Selection Data**

                Choose the lasso or rectangle tool in the graph's menu
                bar and then select points in the graph.

                Note that if `layout.clickmode = 'event+select'`, selection data also
                accumulates (or un-accumulates) selected data if you hold down the shift
                button while clicking.
            """),
            html.Pre(id='selected-data', style=styles['pre']),
        ], className='three columns'),

        html.Div([
            dcc.Markdown("""
                **Zoom and Relayout Data**

                Click and drag on the graph to zoom or click on the zoom
                buttons in the graph's menu bar.
                Clicking on legend items will also fire
                this event.
            """),
            html.Pre(id='relayout-data', style=styles['pre']),
        ], className='three columns')
    ])
])

def get_figure(df, x_col, y_col, selectedpoints, bounds):

    if len(selectedpoints) == 0:
      selectedpoints = []
    # set which points are selected with the `selectedpoints` property
    # and style those points with the `selected` and `unselected`
    # attribute. see
    # https://medium.com/@plotlygraphs/notes-from-the-latest-plotly-js-release-b035a5b43e21
    # for an explanation

    fig = px.scatter(df, x=df[x_col], y=df[y_col],text = df["labels"])

    fig.update_traces(
        selectedpoints=selectedpoints,
        customdata=df.index,
        mode="markers+text",
        marker={"color": "rgba(0, 116, 217, 0.7)", "size": 20},
        unselected={
            "marker": {"opacity": 0.3},
            "textfont": {"color": "rgba(0, 0, 0, 0)"},
        },
    )

    if len(selectedpoints) != 0:
      fig.update_xaxes(
      range=[bounds["x0"],bounds["x1"]],  # sets the range of xaxis
      constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
      )
      fig.update_yaxes(
      range=[bounds["y0"],bounds["y1"]],  # sets the range of xaxis
      constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
      )
    """
    fig.update_layout(
        margin={"l": 20, "r": 0, "b": 15, "t": 5},
        dragmode="select",
        hovermode=False,
        newselection_mode="gradual",
    )
    """
    """
    fig.add_shape(
        dict(
            {"type": "rect", "line": {"width": 1, "dash": "dot", "color": "darkgrey"}},
            **selection_bounds
        )
    )
    """
    return fig


# as a function of the intersection of their 3 selections
@callback(
    Output("basic-interactions", "figure"),
    Output('relayout-data', 'children'),
    Output('intermediate-value', 'data'),
    Input("basic-interactions", "relayoutData"),
    Input('intermediate-value', 'data')
)
def callback_1(selection1,aux):
    try:
      x0,x1,y0,y1 = selection1["xaxis.range[0]"],selection1["xaxis.range[1]"],selection1["yaxis.range[0]"],selection1["yaxis.range[1]"]
      selectedpoints = (df.query('x >= @x0 and x <= @x1 and y >= @y0 and y <= @y1')).index
    except:
      bounds = {"x0": "nel", "x1": "nel", "y1" : "nel", "y0": "nel"}
      datos_extra = json.dumps({"actualizaciones":0}, indent=2)
      #aux = json.dumps(datos_extra, indent=2)
      return  [px.scatter(df, x=df["x"], y=df["y"]), json.dumps(bounds, indent=2), datos_extra]
    #selectedpoints = df.index
    bounds = {"x0": x0, "x1": x1, "y1" : y1, "y0": y0}
    aux_decod = json.loads(aux)
    aux_decod["actualizaciones"] +=1
    return  [get_figure(df, "x", "y", selectedpoints,bounds), json.dumps(bounds,  indent=2), json.dumps(aux_decod,indent=2)]


@callback(
    Output('selected-data', 'children'),
    Input('basic-interactions', 'selectedData'),
    Input('intermediate-value', 'data'))
def display_selected_data(selectedData,aux):
    return json.dumps(aux, indent=2)




if __name__ == "__main__":
    app.run(debug=True, port=8054)