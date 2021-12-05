import os
import cv2
import tempfile
import requests

import torch
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# for training
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in tqdm(val_loader)]
    return model.validation_epoch_end(outputs)


# for calculating the accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


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


def predict(model, image, transform, device):
    transformed = transform(image=image)
    image_tensor = transformed['image']
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))
    probs = torch.softmax(logits, 1).numpy()[0]
    return probs


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