import os
import cv2
import tempfile

import requests
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def read_image_from_url(image_link):
    with tempfile.TemporaryDirectory() as tmpdirname:
        assert os.path.isdir(tmpdirname)
        response = requests.get(image_link)
        assert response.status_code == 200
        file_name = os.path.basename(image_link)
        file_path = os.path.join(tmpdirname, file_name)
        with open(file_path, "wb") as file:
            file.write(response.content)
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def visualize_prediction(image, probs, labels_names):
    H, W, _ = image.shape
    max_new_dim = 300
    scale_factor = max(H, W) / max_new_dim
    H_new = int(H / scale_factor)
    W_new = int(W / scale_factor)
    img_resize = cv2.resize(image, (W_new, H_new))
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Image(z=img_resize), row=1, col=1)
    fig.add_trace(go.Bar(x=labels_names, y=probs), row=1, col=2)
    fig.update_layout(height=500, width=800, title_text="Model predictions", showlegend=False)
    return fig
