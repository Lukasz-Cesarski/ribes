import os

import dash
import torch
import plotly.graph_objects as go
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

from app.utils import read_image_from_url, visualize_prediction, predict
from ribes.nnpp.nn import transforms_valid, idx2cname

LINK_PLACEHOLDER = "Paste link here"
app = dash.Dash(__name__)
server = app.server

device = torch.device("cpu")
model_path = "app/plant-disease-model-complete.pth"
assert os.path.isfile(model_path), model_path
model = torch.load(model_path, map_location=device)
model.eval()


app.layout = html.Div([
    html.H1("Ribes Technologies"),
    html.H3("Apple Leaf Disease Detection Algorithm"),
    html.Div([
        "Image Link: ",
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
        probs = predict(model, image, transforms_valid, device)
        labels_names = [idx2cname[idx] for idx in range(len(idx2cname))]
        fig = visualize_prediction(image, probs, labels_names)

    return [f'Pasted link: {input_link}', f"Counter: {n_clicks}", fig]


if __name__ == "__main__":
    app.run_server(debug=True)
