import os

import dash
import torch
import plotly.graph_objects as go
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from cv2 import error

from app.utils import read_image_from_url, visualize_prediction, predict
from ribes.nnpp.nn import transforms_valid, idx2cname


EXAMPLE_LINK = "https://ag.umass.edu/sites/ag.umass.edu/files/fact-sheets/images/apple-scab_02a.jpg"
SEARCH_SCAB = "https://www.google.pl/search?q=apple+leaf+scab&tbm=isch&ved=2ahUKEwi5_duKhM30AhURxSoKHZAJBdIQ2-cCegQIABAA&oq=apple+leaf+scab"
SEARCH_RUST = "https://www.google.pl/search?q=apple+leaf+rust&tbm=isch&ved=2ahUKEwiF-8vwg830AhUOvyoKHQGjBSwQ2-cCegQIABAA&oq=apple+leaf+rust&gs_lcp"
SEARCH_HEALTHY = "https://www.google.com/search?q=apple+leaf+&tbm=isch&ved=2ahUKEwjr9sSSjM30AhVPx4sKHQvfCkgQ2-cCegQIABAA&oq=apple+leaf"

app = dash.Dash(__name__)
server = app.server

device = torch.device("cpu")
model_path = "app/plant-disease-model-complete.pth"
assert os.path.isfile(model_path), model_path
model = torch.load(model_path, map_location=device)
model.eval()


app.layout = html.Div([
    html.H1("Ribes Technologies"),
    dcc.Link('https://www.ribestech.com/', href='https://www.ribestech.com/', refresh=True),
    html.P("Email: contact@ribestech.com"),
    html.H3("Apple Leaf Disease Detection Algorithm"),
    html.Div([
        "Image Link: ",
        dcc.Input(id='input-link', value=EXAMPLE_LINK, type='text', size="70"),
        html.Button(id='submit-button', n_clicks=0, children='Submit'),
    ]),
    html.Br(),
    dcc.Graph(id='prediction'),
    html.Div(id='output-link'),
    dcc.Link('search HEALTHY images', href=SEARCH_RUST, refresh=True),
    html.Br(),
    dcc.Link('search RUST images', href=SEARCH_RUST, refresh=True),
    html.Br(),
    dcc.Link('search SCAB images', href=SEARCH_SCAB, refresh=True),
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
    try:
        image = read_image_from_url(input_link)
        probs = predict(model, image, transforms_valid, device)
        labels_names = [idx2cname[idx] for idx in range(len(idx2cname))]
        fig = visualize_prediction(image, probs, labels_names)
    except error:
        fig = go.Figure()
        fig.update_layout(title_text="Image downloading failed. Make sure you pasted link to the image source")
    except:
        fig = go.Figure()
        fig.update_layout(title_text=f"Can not connect to page source page.")

    return [f'Input link: {input_link}', f"Request Counter: {n_clicks}", fig]


if __name__ == "__main__":
    app.run_server(debug=True)
