# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 03:55:07 2024

@author: edgar
"""


from dash import Dash, dcc, html, Input, Output, callback, State, no_update
import numpy as np
import pandas as pd
import plotly.express as px
import dash_ag_grid as dag
import base64
import io
from io import StringIO
from dash.exceptions import PreventUpdate

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# la figura inicial es todo blanco
# aunque recordemos que cuando se carga la página, se hace un llamado en el callback
# que tenga como input el componente dcc.Graph
fig = px.scatter()
fig.update_layout(plot_bgcolor='white')
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

# Lo ideal sería no definir esta variable global. Más bien hacer una aplicación
# multipágina y que los resultados de la primera se pasen a la segunda página.
# La otra opción es  almacenar esta infromación en 
#df = pd.DataFrame()




# Los dropdwons donde eliges las opciones para el despliegue de 
# los datos
dropdown_xaxis =  dcc.Dropdown(id = "DP_xaxis", multi=False)
dropdown_yaxis =  dcc.Dropdown(id = "DP_yaxis", multi=False)
dropdown_labels =  dcc.Dropdown(id = "DP_labels", multi=False)

# Upload component
upload = dcc.Upload(
    id='upload_data',
    children=html.Div([
        'Drag and Drop or ',
        html.A('Select Files')
    ]),
    style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px'
    },
    # Allow multiple files to be uploaded
    multiple=False
)


# Con el grid se muestran las entradas de la tabla de los elementos selecionados
# en el plot


grid = dag.AgGrid(
    id="selected_data",
    rowData= []
)

app.layout = html.Div([
    upload,
    html.Div([
    "Selecciona la columna que representa el eje X",
    dropdown_xaxis
    ],id = "x_select"),
    html.Div([
    "Selecciona la columna que representa el eje Y",
    dropdown_yaxis
    ], id = "y_select"),
    html.Div([
    "Selecciona la columna de las etiquetas",
    dropdown_labels
    ], id = "labels_select"),
    html.Button('Graficar', id='graficar_boton'),
    html.Div("",id='mensaje_error',),
    dcc.Graph(
        id='basic-interactions',
        figure= fig,
        config={
            'modeBarButtonsToRemove': ['lasso2d','autoScale2d','zoom2d']
        }
    ),
  dcc.Store(id='estado'),
  dcc.Store(id='accion_anterior'),
  dcc.Store(id='punto_central_anterior'),
  dcc.Store(id = 'xylabels_names'),
  dcc.Store(id = "data_frame"),
  grid
])

def get_figure(df, x_col, y_col, labels_col, selectedpoints, bounds,estado):
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
        True si es un dato marcado y False si no lo es.
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
    if len(selectedpoints)>0:
        new_estado = len(df[x_col])*[False]
        for k in selectedpoints:
            new_estado[k] = True
    else:
        new_estado = estado
        selectedpoints = np.where(np.array(new_estado) == True)[0].tolist()
    texto = [(new_estado[i])*str(df[labels_col][i])+ (1-new_estado[i])*"" for i in range(len(new_estado))]    
    
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
    fig.update_layout(title="DOR and TSNE",plot_bgcolor='white',dragmode ="zoom")
    
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
    autorange = False,
    #constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
    visible = False
    )
    fig.update_yaxes(
    range=[bounds["y0"],bounds["y1"]],  # sets the range of xaxis
    autorange =  False,
    #constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
    visible = False
    )
    fig.add_shape(
        # Rectangle with reference to the plot
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=0,
            x1=1.0,
            y1=1.0,
            line=dict(
                color="black",
                 width=1,
             )
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

def get_initial_values(N):
    """
    Parameters
    ----------
    N : int
        número de datos

    Returns
    -------
    regresa los tres valores: selección, estado y accion_anterior iniciales

    """
    selection= {"autosize": True}
    estado = N*[False]
    accion_anterior = "reset"
    return selection, estado, accion_anterior

def reset_figure(df, x_col,y_col):
    fig = px.scatter(df, x= x_col, y= y_col)
    fig.update_layout(title="DOR and TSNE",plot_bgcolor='white')
    fig.update_traces(marker=dict(size=1,line=dict(width=1,color='blue')))
    #fig.update_layout(clickmode='event')
    xmin, xmax, ymin,ymax = min(df[x_col]),max(df[x_col]),min(df[y_col]),max(df[y_col])
    w,h = xmax-xmin,ymax-ymin
    fig.update_xaxes(range = [xmin - .1*w, xmax +.1*w],visible = False)
    fig.update_yaxes(range = [ymin - .1*h, ymax +.1*h],visible = False)
    fig.add_shape(
        # Rectangle with reference to the plot
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=0,
            x1=1.0,
            y1=1.0,
            line=dict(
                color="black",
                 width=1,
             )
         )
    
    return fig, get_central_point(xmin - .1*w, xmax +.1*w,ymin - .1*h, ymax +.1*h)


def get_central_point(x0,x1,y0,y1):
    return (round(x0 + (x1-x0)/2,5), round(y0 + (y1-y0)/2))

def get_selected_row_data(df,estado):
    #df_selected = df.loc[[i for i in range(estado) if estado[i] == True],["x","y","labels"]]
    df_selected = df.iloc[[i for i in range(len(estado)) if estado[i] == True],:]
    return df_selected.to_dict("records")
    #return [{"x":1,"y":0,"labels":"puto"}]

# Con esta función pasams de la infromación del upload componente a 
# un data frame
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
    return df

# callback para actualizar la imagen
@callback(
    Output("basic-interactions", "figure", allow_duplicate=True),
    Output('estado', 'data'),
    Output('accion_anterior', 'data'),
    Output('punto_central_anterior','data',allow_duplicate=True),
    Output('selected_data', 'rowData'),
    Input("basic-interactions", "relayoutData"),
    Input('estado', 'data'),
    Input('accion_anterior', 'data'),
    Input('punto_central_anterior','data'),
    State('basic-interactions','figure'),
    State('xylabels_names','data'),
    State('data_frame','data'),
    prevent_initial_call=True
)
def callback_1(selection1,estado_entrada,accion_entrada,punto_central_entrada,fig_entrada,xylabels_names,df_codificado):
    
    # En este caso, el usuario aún no proporciona la infromación.
    if xylabels_names is None:
        raise PreventUpdate
    xname,yname,labelsname = xylabels_names[0],xylabels_names[1],xylabels_names[2]
    df = pd.read_json(StringIO(df_codificado),orient="split")
    
    # es como un constructor, cuando se inicia la app se entra en este caso
    if selection1 is None or estado_entrada is None or accion_entrada is None:
        a , estado_salida, accion_salida = get_initial_values(len(df[xname]))
        fig,punto_central_salida = reset_figure(df, xname, yname)
        selected_row_data = []
        return fig, estado_salida, accion_salida, punto_central_salida, selected_row_data
    
    #cuando la acción anterior es desconocida, se reinicia la figura, los estados y la acción
    #a sus valores por default
    if accion_entrada == "unk":
        a, estado_salida, accion_salida = get_initial_values(len(df[xname]))
        fig, punto_central_salida = reset_figure(df, xname, yname)
        selected_row_data = []
        return fig, estado_salida, accion_salida, punto_central_salida, selected_row_data
    
    
    
    # En este caso, actualizamos la figura manualmente y los estados, únicamente cuando
    # la acción anterior fue select o reset, pero además, como maroma para poder distinguir cuando
    # se hizo zoom in y out, verificamos el punto central de la figura anterior.
    if "xaxis.range[0]" in selection1.keys() and "yaxis.range[0]" in selection1.keys():
        accion_salida = accion_entrada
        x0,x1,y0,y1 = selection1["xaxis.range[0]"],selection1["xaxis.range[1]"],selection1["yaxis.range[0]"],selection1["yaxis.range[1]"]
        #selectedpoints = list((df.query('@xname >= @x0 and @xname <= @x1 and @yname >= @y0 and @yname <= @y1')).index)
        selectedpoints = list((df.query('@df[@xname] >= @x0 and @df[@xname] <= @x1 and @df[@yname] >= @y0 and @df[@yname]  <= @y1')).index)
        bounds = {"x0": x0, "x1": x1, "y1" : y1, "y0": y0}
        punto_central_salida = get_central_point(x0, x1, y0, y1)
        if (accion_entrada == "select" or accion_entrada == "reset" or accion_entrada == "zoominout") and punto_central_salida[0] != punto_central_entrada[0] and punto_central_salida[1] != punto_central_entrada[1]:
            fig_salida, estado_salida = get_figure(df, xname, yname, labelsname, selectedpoints, bounds, estado_entrada)
            selected_row_data = get_selected_row_data(df, estado_salida)
            return fig_salida, estado_salida, accion_salida, punto_central_salida, selected_row_data
        elif punto_central_salida[0] == punto_central_entrada[0] and punto_central_salida[1] == punto_central_entrada[1]:
            accion_salida = "zoominout" 
            return no_update, estado_entrada,accion_salida,   punto_central_salida, no_update
    # este caso se tiene que añadir solo por el hecho de que los gráficos funcionan un poco raro.
    # después de haber hecho pan, si se quiere seleccionar de nuevo, entonces se entra en
    # este caso particular
    elif "selections" in selection1.keys():
        accion_salida = accion_entrada
        if len(selection1["selections"]) > 0:
            # por alguna razón las coordenadas de selection1 no cumplen con xk < xj si k < j.
            if "x0" in selection1["selections"][0].keys() and "y0" in selection1["selections"][0].keys():
                x0 = min((selection1["selections"][0]["x0"],selection1["selections"][0]["x1"]))
                x1 = max((selection1["selections"][0]["x0"],selection1["selections"][0]["x1"]))
                y0 = min((selection1["selections"][0]["y0"],selection1["selections"][0]["y1"]))
                y1 = max((selection1["selections"][0]["y0"],selection1["selections"][0]["y1"]))
                punto_central_salida = get_central_point(x0, x1, y0, y1)
                #x0,x1,y0,y1 = selection1["selections"][0]["x0"],selection1["selections"][0]["x1"],selection1["selections"][0]["y0"],selection1["selections"][0]["y1"]
                selectedpoints = list((df.query('@df[@xname] >= @x0 and @df[@xname] <= @x1 and @df[@yname] >= @y0 and @df[@yname]  <= @y1')).index)
                bounds = {"x0": x0, "x1": x1, "y0": y0, "y1" : y1}
                fig_salida, estado_salida = get_figure(df, xname, yname, labelsname, selectedpoints, bounds, estado_entrada)
                selected_row_data = get_selected_row_data(df, estado_salida)
                return fig_salida, estado_salida, accion_salida,  punto_central_salida, selected_row_data
    # En estos casos se acualtiza la acción a una nueva
    elif "dragmode" in selection1.keys():
        if selection1["dragmode"] == "pan":
            accion_salida = "pan"
        if selection1["dragmode"] == "select":
            accion_salida = "select"
            #return fig_entrada, estado_entrada, accion_salida, json.dumps(accion_salida)
            #raise PreventUpdate
    # cuando se haya hecho autosize o ajuste de ejes se reinicia la figura
    elif "xaxis.autorange" in selection1.keys() or "autosize" in selection1.keys():
        fig, punto_central_salida = reset_figure(df, xname, yname)
        a , estado_salida, accion_salida = get_initial_values(len(df[xname]))
        selected_row_data = []
        return fig, estado_salida, accion_salida,  punto_central_salida, selected_row_data
    # si no se tuvo ninguno de los casos anteriores, la acción de salida será desocnocida
    else: 
        accion_salida = "unk"
    estado_salida = estado_entrada
    
    return no_update, estado_salida,accion_salida,  punto_central_entrada, no_update


# callback del boton para graficar. Se verifica que se cumplan las condiciones
# y si es así, se inicializa el despliegue del gráfico y del grid (componente
# para mostrar los datos de los puntos seeccionados).
@callback(
          Output('data_frame', 'data'),
          Output('xylabels_names', 'data'),
          Output('punto_central_anterior','data'),
          Output('selected_data', 'columnDefs'),
          Output('basic-interactions', 'figure'),
          Output('mensaje_error', 'children'),
          Output('mensaje_error', 'style'),
              Output('upload_data', 'style'),
              Output('x_select', 'style'),
              Output('y_select', 'style'),
              Output('labels_select', 'style'),
              Output('graficar_boton','style'),
              Input('graficar_boton','n_clicks'),
              State('DP_xaxis','value'),
              State('DP_yaxis','value'),
              State('DP_labels','value'),
              State('upload_data', 'contents'),
              State('upload_data', 'filename'),
              State('upload_data', 'last_modified')
              )
def submit_button(boton, DPX,DPY,DPL,contents,name,date):
    if DPX is None or DPY is None or DPL is None or contents is None:
        return no_update, no_update,no_update, no_update, no_update, 'No se ha llenado alguno de los campos', no_update, no_update, no_update, no_update, no_update,no_update
    
    try:
        df = parse_contents(contents, name, date)
        df = df.reindex([DPX,DPY,DPL], axis=1)
        fig, punto_medio  = reset_figure(df,DPX,DPY)
        columnDefs = [{'field': str(col)} for col in df.keys()]
        xylabels = [DPX,DPY,DPL]
    except:
        return no_update, no_update,no_update, no_update,no_update, 'Campos no aptos para desplegar un gráfico', no_update, no_update, no_update, no_update, no_update,no_update
    
    return df.to_json(orient="split") ,xylabels, punto_medio , columnDefs, fig, "", {"display": "none"},{"display": "none"},{"display": "none"},{"display": "none"},{"display": "none"},{"display": "none"}





@callback(
              Output('DP_xaxis', 'options'),
              Output('DP_yaxis', 'options'),
              Output('DP_labels', 'options'),
              Input('upload_data', 'contents'),
              Input('upload_data', 'filename'),
              Input('upload_data', 'last_modified')
)
def inicializar_drop_downs(contents,name,date):
    if contents is not None:
        df = parse_contents(contents, name, date)
        opciones_columna = df.columns
        #opciones_columna = ["x","y","z"]
        salida = [{'label': value, 'value': value} for value in opciones_columna]
        return salida,salida,salida
    else:
        raise PreventUpdate()

    



if __name__ == "__main__":
    app.run_server()
