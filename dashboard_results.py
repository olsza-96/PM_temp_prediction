import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import base64
import plotly.graph_objects as go


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

raw_data = pd.read_csv('pmsm_temperature_data.csv')
full_data = pd.read_csv('modelling_results.csv')
errors = pd.read_csv('prediction_errors.csv')

available_measurements = full_data['profile_id'].unique()

def read_description():
    file = open('description.txt', 'r')
    lines = file.read().splitlines()
    file.close()

    return lines


#reading images
heatmap_filename = 'heatmap.png' # path to image
encoded_heat = base64.b64encode(open(heatmap_filename, 'rb').read())


def generate_table(dataframe):
    return html.Table([
        html.Thead(
            html.Tr([html.Th('Feature') for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(col) for col in dataframe.columns
            ]) for i in range(1)])
    ])

def errors_table(dataframe, max_rows=15):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(round(dataframe.iloc[i][col],3)) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    children=[
    html.H1(children='Project by Mateusz S'),

    html.Div( children = [
    html.P('Dashboard presenting results for modelling of permament magnet temperature in the electric motors'),
        html.P('The data used in the project was downloaded from:'),
        html.A('Data source', href = 'https://www.kaggle.com/wkirgsn/electric-motor-temperature?fbclid=IwAR2Ssofcf7beKCcRIm6q_ZEXTwEfGJOsb_0wElXx0fJhLyOwhAJhxMfVkdY', target= '_blank'),
        html.Br()]
    ),

    html.Div([
        html.H4(children='Initial features contained in the dataset:'),
        html.Ul([html.Li(x) for x in read_description()]),
        html.Br()
              ]),

    html.H4(children='Histogram of chosen parameter'),

    html.Div([dcc.RadioItems(
        id = 'radio-hist',
        options=[{'label': i, 'value': i} for i in raw_data.columns],
        value='pm',
        labelStyle={'display': 'inline-block'})],
    ),

    html.Div([
        dcc.Graph(id='hist')
    ], style={'width': '100%', 'display': 'inline-block', 'padding': '0 20'}),

    html.Div(children=
    [
        html.P('Below the heatmap of input features is presented'),
        html.Img(src='data:image/png;base64,{}'.format(encoded_heat.decode()))
    ]),

    html.H4(children='Features chosen after data analysis and feature engineering'),
        generate_table(full_data.drop(columns=['pm', 'profile_id', 'predicted'])),

    html.P('Most of the features have been disregarded due to the fact that they have been found to be irrelevant in the process of modelling'),


    html.H3(children='''
            Visualization of linear regression model results '''),

    html.Div(children='''
        Visualization of permament magnet temperature based on torque, current and voltage measurements 
    '''),
    html.Div([
        dcc.Dropdown(
            id='menu',
            options=[{'label': i, 'value': i} for i in available_measurements],
            value=4
        ),
    ]),
    html.Div([
        dcc.Graph(id='data-id'),
        dcc.Graph(id='model-result')
    ], style={'width': '100%', 'display': 'inline-block', 'padding': '0 20'}),


    html.H4(children='Error results for each measurement ID'),
    html.P('The following columns are:'),
    html.P('MAE - Mean Absolute Error'),
    html.P('MSE - Mean Squared Error'),
    html.P('RMSE - Root Mean Squared Error'),
    html.P('cvRMSE - Coefficient of Variation of RMSE '),
        errors_table(errors)


])


@app.callback(
    dash.dependencies.Output('hist', 'figure'),
    [dash.dependencies.Input('radio-hist', 'value')])
def update_graph(value):
    return histogram(raw_data[value], value)
def histogram(dff, value):
    return {
            'data': [
                {
                    'x': dff,
                    'name': value,
                    'type': 'histogram'
                },
            ],
            'layout': {
                'title': 'Histogram for chosen feature ' + str(value),
                'xaxis':{
                    'title':'Value for chosen feature'
                },
                'yaxis':{
                     'title':'Frequency'
            }
        }}

@app.callback(
    dash.dependencies.Output('data-id', 'figure'),
    [dash.dependencies.Input('menu', 'value')])
def update_graph(value):
    dff = full_data[full_data['profile_id'] == value]
    return create_graph(dff)

def create_graph(dff):
    return {
        'data': [
            {'y': dff.pm, 'type': 'line', 'name': 'Real temperature'},
            {'y': dff.predicted, 'type': 'line', 'name': 'Predicted temperature'}

        ],
        'layout': {
            'title': 'Permament magnet temperature prediction'
        }
    }

@app.callback(
    dash.dependencies.Output('model-result', 'figure'),
    [dash.dependencies.Input('menu', 'value')])
def update_graph(value):
    dff = full_data[full_data['profile_id'] == value]
    return create_correlation(dff)
def create_correlation(dff):
    return {
        'data': [ go.Scatter(x= dff.pm, y= dff.predicted, mode='markers')
        ],
        'layout': {
            'title': 'Correlation of built model',
            'xaxis':{
                    'title': 'Real temperature'
                     },
            'yaxis': {
                    'title': 'Predicted temperature'
        }
    }}


if __name__ == '__main__':
    app.run_server(debug=True)