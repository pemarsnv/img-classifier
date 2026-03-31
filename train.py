import torch
from torch import nn
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import RandomAffine
from torchvision.transforms import ColorJitter
from torchvision.transforms import Compose
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models

import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np

def print_progress(ce, te):
    
    tenth = int(te/10)
    str = ""
    nb = int(ce/tenth)
    for i in range(nb):
        str+="■"
    for i in range(10-nb):
        str+="□"
    print(str+f" {ce}/{te}")
    
def initializeParameters(model, layer3, layer2):
    
    for param in model.parameters(): param.requires_grad = False
    for param in model.fc.parameters(): param.requires_grad = True
    for param in model.layer4.parameters(): param.requires_grad = True

    if (layer3) :
        for param in model.layer3.parameters(): param.requires_grad = True
    if (layer2) :
        for param in model.layer2.parameters(): param.requires_grad = True
 
def buildOptimizer(model, layer3, layer2):

    params = [
        {"params": model.fc.parameters(), "lr": 0.0001},
        {"params": model.layer4.parameters(), "lr": 0.00001}]
    
    if (layer3) : params.append({"params": model.layer3.parameters(), "lr": 0.00001})
    if (layer2) : params.append({"params": model.layer2.parameters(), "lr": 0.000005})

    return torch.optim.Adam(params)

train_transform = Compose([
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    RandomAffine(degrees=0, translate=(0.05, 0.05)), 
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])

eval_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])

### DATA IMPORTATION

train_dataset = ImageFolder("cv_hw1_data/data/train", 
    transform = train_transform, allow_empty = False)

train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

eval_dataset = ImageFolder("cv_hw1_data/data/val", 
    transform = eval_transform, allow_empty = False)

eval_dataloader = DataLoader(eval_dataset, batch_size = 32, shuffle = False)

### CHECK IF MODEL ALREADY EXISTS 

layer3, layer2, cpFound = False, False, False

if (os.path.isfile("checkpoint.pt")):
    
    cpFound = True
    
    print("A model has already been created, trained and saved.")
    print("Do you want to train this process again? (Y/N)")
    ans = input()
    
    if (ans == "Y"): 
        
        ### Choose which layer to work on this iteration 
        
        print("Do you want to unfreeze layer 3?")
        
        ans = input()
        layer3 = False
        if (ans == "Y"): 
            layer3 = True
            print("Layer 3 unfreezed")
        else:
            print("Layer 3 freezed")
        
        print("Do you want to unfreeze layer 2?")
        
        ans = input()
        layer2 = False
        if (ans == "Y"): 
            layer2 = True
            print("Layer 2 unfreezed")
        else: 
            print("Layer 2 freezed")
    
    else:
        
        print("Training cancelled. Exiting program.")
        sys.exit(0)
        
    checkpoint = torch.load("checkpoint.pt")
    epoch_num = int(checkpoint["epoch"])
    
else: 
    print("Creating new model.")
    epoch_num = 0
    

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
if cpFound: model.load_state_dict(checkpoint["model"])
initializeParameters(model, layer3, layer2)

optimizer = buildOptimizer(model, layer3, layer2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.1,
    patience=3
)

if cpFound: 
    layers = checkpoint["layers"]
    if (layers["layer3"] == layer3 and layers["layer2"] == layer2):
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    
print("How many epochs do you want to execute?")
ans = input()
digit = ans.isdigit()

while (not digit):
    
    print("You need to input a digit")
    print("How many epochs do you want to execute?")
    ans = input()
    digit = ans.isdigit()
    
nb_epoch = int(ans)

print("Training starting...")

total = len(train_dataloader)
current = 0

losses = []
accuracies = []

loss_evaluator = nn.CrossEntropyLoss()

for i in range(epoch_num, epoch_num + nb_epoch):
    
    print(f"INITIATING EPOCH N°{i+1}")
    print(f"□□□□□□□□□□ {current}/{total}")

    ### Training phase
    
    model.train()
    
    for (images, labels) in train_dataloader:
        
        optimizer.zero_grad()
        
        output = model(images)
        loss = loss_evaluator(output, labels)
        loss.backward()
        
        current+=1
        
        if current % 32 == 0: print_progress(current, total)
        
        optimizer.step()
        
    print_progress(current, total)
    current=0
    
    ### Evaluation phase
    
    model.eval()
    loss_value, nb_correct, nb_total = 0, 0, 0
    
    for (images, labels) in eval_dataloader:
        
        output = model(images)
        loss = loss_evaluator(output, labels)
        
        loss_value += loss.item()
        
        predicted = torch.max(output, 1)[1]
        nb_correct += (predicted == labels).sum().item()
        nb_total += labels.size(0)
        
    accuracy = nb_correct / nb_total
    
    print(f"EPOCH N°{i+1} COMPLETE.")
    print(f"Loss: {loss_value:.4f} | Accuracy: {accuracy:.4f}")
    
    losses.append(loss_value)
    accuracies.append(accuracy)
    
    scheduler.step(accuracy)
    
epoch_num += nb_epoch    
    
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "layers": {
        "layer3": layer3,
        "layer2": layer2
    },
    "epoch": epoch_num
}, "checkpoint.pt")
    
x = range(epoch_num, len(losses) + epoch_num)
np_losses = np.array(losses)
np_accuracies = np.array(accuracies)

plt.subplot(1,2,1)
plt.plot(x, np_losses)
plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(x, np_accuracies)
plt.title("Accuracy")

plt.show()

print("Training session over, saving model.")