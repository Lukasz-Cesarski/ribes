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
    html.Link(
        rel='stylesheet',
        href='/assets/main.css'
    ),
    html.Link(
        rel='stylesheet',
        href='/assets/boostrap.css'
    ),
    html.Link(
        rel='stylesheet',
        href='/assets/font-awesome.css'
    ),
    html.Header(html.A(html.Strong("Ribes Technologies"), className="logo"), id="header", className="mb-5"),
    html.Div(className="mt-5"),
    html.Section(
      html.Section(
        html.Article(
          [
            html.Span(
              html.Img(src="/assets/images/pic14.jpg"), className="image", style="display: none"
            ),
            html.Header(
              html.H3("Ribes Technologies - Apple Leaf Disease Detection Algorithm")
            )
          ], style={ "background-image": "url('/assets/images/pic14.jpg')" }, className="w-100",
        ), className="tiles"
      ), className="spotlights"
    ),
    html.Div(className="mt-5"),
    html.Div([
      html.H2("Paste link to photo of a leaf: ", className="mb-2"),
      dcc.Input(id='input-link', value=EXAMPLE_LINK, type='text', size="70"),
      html.Div(
        html.Div(
          html.Button(id='submit-button', n_clicks=0, children='Submit'), className="col d-flex justify-content-center"
        ), className="row justify-content-md-center mt-3"
      )
    ]),
    html.Br(),
    dcc.Graph(id='prediction'),
    html.Div(id='output-link'),
    html.Div(className="mt-5"),
    html.Div(
      [
        html.Div(dcc.Link('search HEALTHY images', href=SEARCH_RUST, refresh=True, className="button"), className="col d-flex justify-content-center"),
        html.Div(dcc.Link('search RUST images', href=SEARCH_RUST, refresh=True, className="button"), className="col d-flex justify-content-center"),
        html.Div(dcc.Link('search SCAB images', href=SEARCH_SCAB, refresh=True, className="button"), className="col d-flex justify-content-center")
      ], className="row d-flex justify-content-around"
    ),
    html.Div(id='n-clicks'),
    html.Div(className="mt-5"),
    html.Div(className="mt-5"),
    html.Section(
      html.Div(
          [
              html.H2("Contact us"),
              html.H3("Email"),
              html.A("ribestech.com"),
              html.Br(),
              html.A("contact@ribestech.com"),
          ], className="contact-method"
      ), id="contact"
    )
], className="container")

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
