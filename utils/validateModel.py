import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from lion_pytorch import Lion

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

class GrayscaleToRgb(object):
    def __call__(self, img):
        return img.expand(3, -1, -1)

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
}

train = datasets.CIFAR100("../data/CIFAR100/", train=True, transform=data_transforms['train'])
train_size = int(0.9 * len(train))
val_size = len(train) - train_size
train, val = torch.utils.data.random_split(train, [train_size, val_size])

test = datasets.CIFAR100("../data/CIFAR100/", train=False, transform=data_transforms['test'])

bz = 32
dataloaders = {
    'train': torch.utils.data.DataLoader(train, batch_size=bz, shuffle=True, num_workers=8),
    'val': torch.utils.data.DataLoader(val, batch_size=bz, shuffle=False, num_workers=8),
    'test': torch.utils.data.DataLoader(test, batch_size=bz, shuffle=False, num_workers=8)
}

dataset_sizes = {
    'train': len(train),
    'val': len(val),
    'test': len(test)
}

class_names = test.classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(dataset_sizes)
print(device)
print(len(class_names), class_names, "\n")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=100)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out


#### ConvNet as fixed feature extractor ####
# model_conv = FashionCNN()
# model_conv = torchvision.models.resnet152()
model_conv = torchvision.models.resnext101_64x4d(weights=torchvision.models.ResNeXt101_64X4D_Weights.DEFAULT)

# for param in model_conv.parameters():
#     param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, len(class_names))
hidden_size = 256
model_conv.fc = nn.Sequential(
    nn.Linear(num_ftrs, hidden_size),
    nn.ReLU(inplace=True),
    nn.Linear(hidden_size, hidden_size//2),
    nn.ReLU(inplace=True),
    nn.Linear(hidden_size//2, len(class_names))
    )

# Change to GPU learning if possible
model_conv = model_conv.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.Adam(model_conv.parameters())
# optimizer_conv = Lion(
#     model_conv.fc.parameters(), 
#     lr = 1e-3,
#     weight_decay = 1e-2
# )

# Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=10, gamma=0.1)

# Train model
model_conv = train_model(model_conv, criterion, optimizer_conv,
                        exp_lr_scheduler, num_epochs=15)

def test_model(model, dataloader):
    corrects = [0] * len(class_names)
    totals = [0] * len(class_names)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in range(len(class_names)):
                mask = labels == i
                corrects[i] += torch.sum(preds[mask] == labels[mask]).item()
                totals[i] += torch.sum(labels == i).item()

    accs = [c / t for c, t in zip(corrects, totals)]
    overall_acc = sum(corrects) / sum(totals)
    print("\n\n")
    print('Overall Test Accuracy: {:.4f}'.format(overall_acc))
    print('-' * 10)
    for i in range(len(class_names)):
        print('{} Accuracy: {:.4f}'.format(class_names[i], accs[i]))
    
    return overall_acc, accs

test_model(model_conv, dataloaders['test'])

