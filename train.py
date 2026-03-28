import torch
from torch import nn
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models

import os.path
import sys

def transform(image):
    resized_img = Resize((224, 224))(image)
    return ToTensor()(resized_img)

def print_progress(ce, te):
    
    if ce == te:
        print(f"■■■■■■■■■■ {ce}/{te} TRAINING COMPLETE.")
    
    tenth = int(te/10)
    print(f"tenth={tenth}")
    str = ""
    nb = int(ce/tenth)
    for i in range(nb):
        str+="■"
    for i in range(10-nb):
        str+="□"
    print(str+f" {ce}/{te}")

### Importing the data 

train_dataset = ImageFolder("cv_hw1_data/data/train", 
    transform = transform, allow_empty = False)

train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

### Check if model already exists 

if (not os.path.isfile("model.pt")):
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 100)
    print("New model successfully created.")
    
else:
    
    print("A model has already been created, trained and saved.")
    print("Do you want to train this process again? (Y/N)")
    ans = input()
    if (ans != "Y"): 
        print("Training cancelled. Exiting program.")
        sys.exit(0)
    model = torch.load("model.pt", weights_only= False)
    print("Model successfully imported.")
    
### Import optimizer and loss function    
    
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

loss_evaluator = nn.CrossEntropyLoss()

model.train()

print("Training starting...")

total_epoch = len(train_dataloader)
current_epoch = 0

print(f"□□□□□□□□□□ {current_epoch}/{total_epoch}")

for (images, labels) in iter(train_dataloader):
    
    output = model(images)
    loss = loss_evaluator(output, labels)
    loss.backward()
    
    current_epoch+=1
    print_progress(current_epoch, total_epoch)

print("Training session over, saving model.")
torch.save(model, "model.pt")
