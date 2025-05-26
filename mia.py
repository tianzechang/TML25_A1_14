import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple
import numpy as np
import requests
import pandas as pd
from scipy.stats import norm

#### LOADING THE MODEL

from torchvision.models import resnet18

### Add this as a transofrmation to pre-process the images
mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]

model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 44)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

ckpt = torch.load("./01_MIA.pt", map_location="cpu")

model.load_state_dict(ckpt)
model.eval()

#### DATASETS
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index): # ?
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index): # ?
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]

public_data: MembershipDataset = torch.load("./pub.pt")
private_data: MembershipDataset = torch.load("./priv_out.pt")

conf_in = []
conf_out = []

with torch.no_grad():
    for i in range(len(public_data)):
        id_, img, label, member = public_data[i]
        img = transform(img).unsqueeze(0)
        output = model(img)
        prob = torch.softmax(output, dim=1)[0, label].item()
        if member == 1:
            conf_in.append(prob)
        else:
            conf_out.append(prob)

mu_in, std_in = np.mean(conf_in), np.std(conf_in) + 1e-8
mu_out, std_out = np.mean(conf_out), np.std(conf_out) + 1e-8

scores = []

with torch.no_grad():
    for i in range(len(private_data)):
        id_, img, label, _ = private_data[i]
        img = transform(img).unsqueeze(0)
        output = model(img)
        prob = torch.softmax(output, dim=1)[0, label].item()
        p_in = norm.pdf(prob, mu_in, std_in)
        p_out = norm.pdf(prob, mu_out, std_out)
        lira_score = p_in / (p_out + 1e-8)
        membership_prob = lira_score / (1 + lira_score)
        scores.append((id_, membership_prob))


#### EXAMPLE SUBMISSION
df = pd.DataFrame({
    "ids": [x[0] for x in scores],
    "score": [x[1] for x in scores]
})
df.to_csv("test.csv", index=None)

