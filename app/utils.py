import os
import cv2
import tempfile
from PIL import Image
import torch
import requests
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import torchvision.transforms as transforms

HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

def read_image_from_url(image_link):
    response = requests.get(image_link, headers=HEADERS)
    assert response.status_code == 200
    with tempfile.TemporaryDirectory() as tmpdirname:
        assert os.path.isdir(tmpdirname)
        file_name = "file"
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
    image_pil = Image.fromarray(image)
    img_resize = transforms.Resize((H_new, W_new))(image_pil)
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Image(z=img_resize), row=1, col=1)
    fig.add_trace(go.Bar(x=labels_names, y=probs), row=1, col=2)
    fig.update_layout(height=500, width=1000, showlegend=False)
    fig.update_yaxes(title_text="Probability", range=[0, 1], row=1, col=2)
    fig.update_xaxes(title_text="Disease", row=1, col=2)
    fig.update_xaxes(title_text="Image preview", row=1, col=1)
    return fig


def predict(model, image, transform, device):
    image_pil = Image.fromarray(image)
    transformed = transform(image_pil)
    image_tensor = transformed

    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))
    probs = torch.softmax(logits, 1).numpy()[0]
    return probs
