import os
import sys

import torch
from torchsummary import summary

from ribes.read_data import get_dataloader
from ribes.utils import to_device, get_default_device
from ribes.nn import ResNet9, fit_OneCycle


if __name__ == "__main__":
    if "KAGGLE_CONTAINER_NAME" in os.environ:
        sys.path.insert(0, "../input/ribes-github")

    batch_size = 2
    device = get_default_device()

    train_ddl, valid_ddl, train_if, valid_if = get_dataloader(batch_size, device)

    model = to_device(ResNet9(3, len(train_if.classes)), device)

    INPUT_SHAPE = (3, 256, 256)
    print(summary(model, INPUT_SHAPE, device=device.type))

    history = []

    epochs = 2
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    history += fit_OneCycle(epochs, max_lr, model, train_ddl, valid_ddl,
                            grad_clip=grad_clip,
                            weight_decay=weight_decay,
                            opt_func=opt_func)
