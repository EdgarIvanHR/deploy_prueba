# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 19:35:58 2024

@author: edgar
"""

from dash import Dash, dcc, html, Input, Output, callback
import numpy as np
import pandas as pd
import plotly.express as px
import ujson as json

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
df = pd.read_csv("DOR_TSNE.csv",index_col="Unnamed: 0")
#texto= [t for t in df["labels"]]
fig = px.scatter(df, x="x", y="y")
fig.update_layout(title="DOR and TSNE",plot_bgcolor='white')
fig.update_traces(marker=dict(size=1,line=dict(width=1,color='blue')))
#fig.update_layout(clickmode='event+select')
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)



app.layout = html.Div([
    dcc.Graph(
        id='basic-interactions',
        figure=fig
    ),
  dcc.Store(id='estado'),
  dcc.Store(id='accion_anterior')
])

def get_figure(df, x_col, y_col, selectedpoints, bounds,estado,accion_anterior):
    """
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    x_col : TYPE
        DESCRIPTION.
    y_col : TYPE
        DESCRIPTION.
    selectedpoints : list
        lista de los indices de los puntos seleccionados
    bounds : TYPE
        DESCRIPTION.
    estado : list
        1 si es un dato marcado y 0 si no lo es.
        En base a esta lista se coloca el texto en los puntos.
        Esta lista se acutaliza según la información en selectedpoints.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    # si se hizo zoom y no se obtuvieron puntos a seleccionar o si
    # la acción anterior fue de un paneo, entonces los puntos
    # seleccionados serán los mismos que ya se tenían previamente.
    # En caso contrario (la sentencia del if == True) se acualizan los
    # puntos seleccionados y el estado.
    if len(selectedpoints)>0 and accion_anterior != "pan":
        new_estado = len(df[x_col])*[False]
        for k in selectedpoints:
            new_estado[k] = True
    else:
        new_estado = estado
        selectedpoints = np.where(np.array(new_estado) == True)[0].tolist()
    texto = [(new_estado[i])*str(df["labels"][i])+ (1-new_estado[i])*"" for i in range(len(new_estado))]    
    
    #texto = len(df["x"])*["puto"]
    # set which points are selected with the `selectedpoints` property
    # and style those points with the `selected` and `unselected`
    # attribute. see
    # https://medium.com/@plotlygraphs/notes-from-the-latest-plotly-js-release-b035a5b43e21
    # for an explanation
    #texto = len(df.index)*["hola"]
    #texto = [str(t) for t in df["labels"]]
    #texto = df["labels"]
    fig = px.scatter(df, x= x_col, y= y_col,text = texto)
    fig.update_layout(title="DOR and TSNE",plot_bgcolor='white')
    
    fig.update_traces(
        selectedpoints=selectedpoints,
        customdata=df.index,
        mode="markers+text",
        marker={"color": "rgba(0, 116, 217, 0.7)", "size": 5},
        #marker=dict(size=1.5,line=dict(width=2,color='blue')),
        textposition= 'top center',
        unselected={
            "marker": {"opacity": 0.3}
            #"textfont": {"color": "rgba(1, 0, 0, 1)"},
        },
    )
    
    #fig.update_layout(clickmode='event+select')
    #if len(selectedpoints) != 0:
    fig.update_xaxes(
    range=[bounds["x0"],bounds["x1"]],  # sets the range of xaxis
    constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
    visible = False
    )
    fig.update_yaxes(
    range=[bounds["y0"],bounds["y1"]],  # sets the range of xaxis
    constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
    visible = False
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
    return fig, new_estado


# as a function of the intersection of their 3 selections
@callback(
    Output("basic-interactions", "figure"),
    Output('estado', 'data'),
    Output('accion_anterior', 'data'),
    Input("basic-interactions", "relayoutData"),
    Input('estado', 'data'),
    Input('accion_anterior', 'data')
)
def callback_1(selection1,estado,accion_entrada):
    if estado is not None :
        estado_salida_cod = estado
    else:
        estado_salida_decod = len(df.index)*[False]
        estado_salida_cod = json.dumps(estado_salida_decod)
    
    if selection1 is None:
        bounds = {"x0": "nel", "x1": "nel", "y1" : "nel", "y0": "nel"}
        #aux = json.dumps(datos_extra, indent=2)
        fig = px.scatter(df, x=df["x"], y=df["y"])
        fig.update_layout(title="DOR and TSNE",plot_bgcolor='white')
        fig.update_traces(marker=dict(size=1,line=dict(width=1,color='blue')))
        #fig.update_layout(clickmode='event+select')
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        estado_salida_decod = len(df.index)*[False]
        estado_salida_cod = json.dumps(estado_salida_decod)
        # se llama accion anterior, pero en realidad en este caso es
        # la acción actual, donde empieza todo. Se inicializa el valor de esta
        #variable a "reinicio" que es el mismo para otra acciones.
        # realmente nuestra principal coupación es con las acciones de "pan" y "select".
        # si nos fijamos, accion anterior es la accion que es retornada en el callback
        #y accion_actual es la que usamos como tal.
        accion_anterior_cod = json.dumps("reinicio")
        return [fig,estado_salida_cod,accion_anterior_cod]
    elif "dragmode" in selection1.keys():
        if selection1["dragmode"] == "pan":
            accion_anterior_cod = json.dumps("pan")
            
        elif selection1["dragmode"] == "select":
            accion_anterior_cod = json.dumps("select")
        else:
            accion_anterior_cod = json.dumps("unk")
    elif "xaxis.range[0]" in selection1.keys():
        x0,x1,y0,y1 = selection1["xaxis.range[0]"],selection1["xaxis.range[1]"],selection1["yaxis.range[0]"],selection1["yaxis.range[1]"]
        selectedpoints = list((df.query('x >= @x0 and x <= @x1 and y >= @y0 and y <= @y1')).index)
        bounds = {"x0": x0, "x1": x1, "y1" : y1, "y0": y0}
        axion_actual_decod = json.loads(accion_entrada)
        estado_actual_decod  = json.loads(estado)
        #fig = px.scatter(df, x=df["x"], y=df["y"])
        #fig.update_layout(title="DOR and TSNE",plot_bgcolor='white')
        fig, estado_salida_decod = get_figure(df, "x", "y", selectedpoints,bounds,estado_actual_decod,axion_actual_decod)
        estado_salida_cod = json.dumps(estado_salida_decod)
        accion_anterior_cod = json.dumps("zoom")
        return  [fig, estado_salida_cod, accion_anterior_cod]
        #return  [get_figure(df, "x", "y", selectedpoints,bounds), json.dumps(list(selectedpoints),indent = 2), json.dumps(aux_decod,indent=2)]
        #return [fig, json.dumps(selection1,indent = 2), json.dumps(aux_decod,indent=2)]
    else:
        accion_anterior_cod = json.dumps("no_interesa")
    


    #aux = json.dumps(datos_extra, indent=2)

    fig = px.scatter(df, x=df["x"], y=df["y"])
    fig.update_layout(title="DOR and TSNE",plot_bgcolor='white')
    fig.update_traces(marker=dict(size=1,line=dict(width=1,color='blue')))
    #fig.update_layout(clickmode='event+select')
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    #return  [fig, json.dumps(bounds, indent=2), datos_extra]
    return [fig, estado_salida_cod, accion_anterior_cod]
    #selectedpoints = df.index
   


if __name__ == "__main__":
    app.run_server(debug = False)
