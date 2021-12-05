import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

from app.utils import read_image_from_url, visualize_prediction


LINK_PLACEHOLDER = "Paste link here"
app = dash.Dash(__name__)
server = app.server

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
    dcc.Graph(id='prediction'),
    html.Div(id='n-clicks'),
])

@app.callback(
    Output(component_id='output-link', component_property='children'),
    Output(component_id='n-clicks', component_property='children'),
    Output(component_id='prediction', component_property='figure'),
    State(component_id='input-link', component_property='value'),
    Input(component_id='submit-button', component_property='n_clicks'),
)
def update_output_div(input_link, n_clicks):
    if input_link == LINK_PLACEHOLDER:
        fig = go.Figure()
    else:
        image = read_image_from_url(input_link)
        probs = [0.25, 0.25, 0.25, 0.25]
        labels_names = ["healthy", "mult. diseases", "rust", "scab"]
        fig = visualize_prediction(image, probs, labels_names)

    return [f'Pasted link: {input_link}', f"Predictions counter: {n_clicks}", fig]


if __name__ == "__main__":
    app.run_server(debug=True)
