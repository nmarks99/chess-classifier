#!/usr/bin/env python3

import sys
import torch
from torch import nn
import torchvision
import torchvision.transforms.functional as fn
from torchvision.transforms import Resize
from PIL import Image
from utils import brint

# Define a neural network as a class that inherits from the torch.nn.Module class 
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.model = torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT")
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Linear(1000, 5)
        )
    def forward(self, x):
        x = self.model(x)
        return x


def import_image(img_path):
    '''
    Imports the image at the provided path and reshapes
    it and converts it to a tensor so it can be evaluated
    by the model
    '''
    img_raw = Image.open(img_path)
    resize_tf= Resize((224,224))
    resized_img = resize_tf(img_raw)
    img = fn.to_tensor(resized_img)
    img.unsqueeze_(0)
    if img.size() == (1,1,224,224):
        img = torch.cat([img, img, img],dim=1)
    assert(img.size() == (1,3,224,224)),"Image resizing failed"

    return img


# Define the data classes
classes = ["Queen","Rook","Bishop","Knight","Pawn"]

# Get path to image from command line args
assert(len(sys.argv) == 2 or len(sys.argv) == 3), "Invalid input"
if len(sys.argv) == 3:
    model_path = sys.argv[2]
else:
    model_path = None
img_path = sys.argv[1]

# Initialize the pre-trained model
model = ChessNet()

if not model_path:
    model_path = "./model_SGD.pt"
model.load_state_dict(torch.load(model_path,map_location=torch.device("cpu")))

# Get the image
img = import_image(img_path)

with torch.no_grad():
    model.eval()
    yp = model(img)
    yp = nn.Softmax(dim=1)(yp)
    top_p, top_class = yp.topk(1, dim=1)
    brint(f"{classes[top_class.item()]}",color="BOLD_GREEN")
    print(f"Confidence = {round(top_p.item() * 100,2)}%")






