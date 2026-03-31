import torch
import os 
from PIL import Image
import csv

from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.transforms import Compose

import torchvision.models as models

transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])])


model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model"])
model.eval()

data = [["image_name","pred_label"]]

for file in sorted(os.listdir("cv_hw1_data/data/test")):
    
    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img = Image.open("cv_hw1_data/data/test/"+file).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)

    output = model(img)
    prediction = torch.max(output, 1)[1].item()

    data.append([file.removesuffix(".jpg"), prediction])

with open("prediction.csv", mode='w', newline='') as file:
    
    writer = csv.writer(file)
    writer.writerows(data)