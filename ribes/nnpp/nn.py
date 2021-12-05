import os
import numpy as np

import albumentations as A
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import roc_auc_score


class PlantDataset(Dataset):

    def __init__(self, df, dir_input, transforms=None):
        self.df = df
        self.transforms = transforms
        self.dir_input = dir_input

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_src = os.path.join(self.dir_input, 'images', self.df.loc[idx, 'image_id'] + '.jpg')
        assert os.path.isfile(image_src), image_src
        # print(image_src)
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']].values
        labels = torch.from_numpy(labels.astype(np.int8))
        labels = labels.unsqueeze(-1)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

        return image, labels


class PlantModel(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        self.backbone = torchvision.models.resnet18(pretrained=True)

        in_features = self.backbone.fc.in_features

        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = F.dropout(x, 0.25, self.training)

        x = self.logit(x)

        return x


class DenseCrossEntropy(nn.Module):

    def __init__(self):
        super(DenseCrossEntropy, self).__init__()

    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.float()

        logprobs = F.log_softmax(logits, dim=-1)

        loss = -labels * logprobs
        loss = loss.sum(-1)

        return loss.mean()


def train_one_fold(i_fold, model, criterion, optimizer, dataloader_train, dataloader_valid, n_epochs, device):
    train_fold_results = []

    for epoch in range(n_epochs):

        # print('  Epoch {}/{}'.format(epoch + 1, N_EPOCHS))
        # print('  ' + ('-' * 20))
        os.system(f'echo \"  Epoch {epoch}\"')

        model.train()
        tr_loss = 0

        for step, batch in enumerate(tqdm(dataloader_train)):
            images = batch[0]
            labels = batch[1]

            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            outputs = model(images)
            loss = criterion(outputs, labels.squeeze(-1))
            loss.backward()

            tr_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

        # Validate
        model.eval()
        val_loss = 0
        val_preds = None
        val_labels = None

        for step, batch in enumerate(dataloader_valid):

            images = batch[0]
            labels = batch[1]

            if val_labels is None:
                val_labels = labels.clone().squeeze(-1)
            else:
                val_labels = torch.cat((val_labels, labels.squeeze(-1)), dim=0)

            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            with torch.no_grad():
                outputs = model(images)

                loss = criterion(outputs, labels.squeeze(-1))
                val_loss += loss.item()

                preds = torch.softmax(outputs, dim=1).data.cpu()

                if val_preds is None:
                    val_preds = preds
                else:
                    val_preds = torch.cat((val_preds, preds), dim=0)

        train_fold_results.append({
            'fold': i_fold,
            'epoch': epoch,
            'train_loss': tr_loss / len(dataloader_train),
            'valid_loss': val_loss / len(dataloader_valid),
            'valid_score': roc_auc_score(val_labels, val_preds, average='macro'),
        })

    return val_preds, train_fold_results


SIZE = 512

transforms_train = A.Compose([
    A.RandomResizedCrop(height=SIZE, width=SIZE, p=1.0),
    A.Flip(),
    A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),

    # Pixels
    A.OneOf([
        A.Emboss(p=1.0),
        A.Sharpen(p=1.0),
        A.Blur(p=1.0),
    ], p=0.5),

    # Affine
    A.OneOf([
        A.ElasticTransform(p=1.0),
        A.PiecewiseAffine(p=1.0)
    ], p=0.5),

    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])

transforms_valid = A.Compose([
    A.Resize(height=SIZE, width=SIZE, p=1.0),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])

idx2cname = {0: 'healthy', 1: 'multiple_diseases', 2: 'rust', 3: 'scab'}
