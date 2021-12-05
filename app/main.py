import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

LINK_PLACEHOLDER = "Paste link here"
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Ribes Technologies"),
    html.H3("Apple Leaf Disease Detection"),
    html.Div([
        "Apple Leaf Image Link: ",
        dcc.Input(id='input-link', value=LINK_PLACEHOLDER, type='text'),
        html.Button(id='submit-button', n_clicks=0, children='Submit'),
    ]),
    html.Br(),
    html.Div(id='output-link'),

])

@app.callback(
    Output(component_id='output-link', component_property='children'),
    State(component_id='input-link', component_property='value'),
    Input(component_id='submit-button', component_property='n_clicks'),
)
def update_output_div(input_link, n_clicks):
    return f'Output: {input_link} n_clicks={n_clicks}'


if __name__ == "__main__":
    app.run_server(debug=True)
