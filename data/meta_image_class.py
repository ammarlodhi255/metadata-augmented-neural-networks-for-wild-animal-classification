import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import transforms
import time
import copy
import json
import torch.utils.data as data
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# self defined functions
from fusion_model import FusionModel, EarlyFusionModel, MetadataModel
from py.Conventional_models import ResNet18Model

# Dataset
from NINA_dataset import NINADataset
from NINA_meta_dataset import NINAMetaDataset

# reduces the size of the dataset, respective to the identifier
def reduce_sample_size(annotations, identifier, new_size):
    """
    Reduces the data size of each individual identifier.

    Reduces the size of the dataset by splitting the dataset into a dictionary
    with each unique identifier as the key, and a list of all datapoints which 
    matches that identifier. Afterwards, each list is reduced to the size 
    len(samples_of_identifier)*new_size. 

    ---------
    Parameters:
    annotations - list of all samples in dataset
    identifier - unique key to split list by, use class label if unsure
    new_size - float in range 0 - 1 determining the % size of the new dataset 1 means no samples are removed.

    ---------
    Returns: size reduced annotations list
    """
    d = {}
    for entry in annotations:
        if(entry[identifier] not in d.keys()):
            d[entry[identifier]] = [entry]
        else:
            d[entry[identifier]].append(entry)
    new_annotations = []
    for key in d.keys():
        samples = d[key]
        sample_size = int(new_size * len(samples))
        discard_size = len(samples) - sample_size
        samples, _ = torch.utils.data.random_split(samples, [sample_size, discard_size])    # type: ignore
        new_annotations += samples
    return new_annotations

def confusion_matrix(model, dataloader):
    n_classes = len(class_names)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    with torch.no_grad():
        for inputs, metadata, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            metadata = metadata.to(device)
            outputs = model(inputs, metadata)
            _, preds = torch.max(outputs, 1)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                try:
                    conf_matrix[t.long(), p.long()] += 1
                except:
                    print(t.long(), p.long())

    return conf_matrix

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1e10

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
            for inputs, metadata, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                metadata = metadata.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, metadata)
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
            epoch_acc = running_corrects.double() / dataset_sizes[phase] # type: ignore

            print('{}; Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if(phase== 'val' and epoch_acc > best_acc):
                best_acc = epoch_acc
            
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f"./weights_{model_iteration}.bin")


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def create_model(model_type, input_size, output_size):
    if model_type == "ResNet18":
        return ResNet18Model(input_size, output_size)
    elif model_type == "Fusion":
        return FusionModel(input_size, output_size)
    elif model_type == "EarlyFusion":
        return EarlyFusionModel(input_size, output_size)
    elif model_type == "MetadataModel":
        return MetadataModel(input_size, output_size)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

# transforms used on data
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])
transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((256, 256)), # only if 512x512 is too large for GPU
        transforms.Normalize(mean, std)
    ])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1)), # type: ignore
])

# Data loading
bp = "/media/user-1/CameraTraps/NINA/Images/"
with open(bp + "metadata.json") as f:
    metadata = json.load(f)

categories = metadata['categories']
annotations = metadata['annotations']

# reduce size of whole dataset for trial runs (range 0-1, lower number means less data)
# annotations = reduce_sample_size(annotations, 'Species_ID', 0.01)



class_names = [cat for cat in categories.keys()]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = len(class_names)
print(device)
print(m, categories)

DL_models = ["ResNet18"]

losses = [nn.CrossEntropyLoss()]

n_splits = 10  # Number of splits for k-fold cross-validation

confusion_matrices = []
models_run = []
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
model_iteration = 0 # used to save each model individually
for _ in range(len(DL_models)*len(losses)):
    confusion_matrices.append(np.zeros((m, m, n_splits)))

for model_string in DL_models:
    print(model_string)
    for criterion in losses:
        for run, (train_index, test_index) in enumerate(kf.split(annotations)):

            # data splitting
            annotations_train = data.Subset(annotations, train_index)
            annotations_train, annotations_val = train_test_split(annotations_train, test_size=0.1)
            annotations_test = data.Subset(annotations, test_index)

            # datasets
            dataset_train = NINADataset(annotations_train, bp, transform=transform, augment_transform=train_transform, augment=True)
            dataset_val = NINADataset(annotations_val, bp, transform=transform)
            dataset_test = NINADataset(annotations_test, bp, transform=transform)

            class_sample_counts = np.array(list(categories.values()))  # Assuming train_labels is a list of species IDs for the training set
            class_weights = 1. / np.array(class_sample_counts, dtype=float)
            
            # using species_ID as the label
            sample_weights = np.array([class_weights[label] for label in np.arange(len(categories.keys()))])
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

            # Create a data loader with the weighted sampler
            bz = 75

            dataloaders = {
                'train': DataLoader(dataset_train, batch_size=bz, sampler=sampler, num_workers=8),
                'val': DataLoader(dataset_val, batch_size=bz, shuffle=False, num_workers=8),       
                'test': DataLoader(dataset_test, batch_size=bz, shuffle=False, num_workers=8) 
            }
            dataset_sizes = {
                'train': len(dataset_train),
                'val': len(dataset_val),
                'test': len(dataset_test)
            }
            _, meta_features, _ = dataset_train[0]
            model = create_model(model_string, len(meta_features), len(class_names))

            model = model.to(device)
            optimizer_conv = optim.Adam(model.parameters(), lr=1e-4)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
            model = train_model(model, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=15)
            
            conf_mat = confusion_matrix(model, dataloaders['test'])
            confusion_matrices[model_iteration][:, :, run] = conf_mat
        models_run.append(model_string)
        model_iteration += 1 # new loss or model => new model_iteration

print(f"Mean results using {n_splits}-fold cross validation")
model_iteration = 0 # used to save each model individually
for (conf_mat, model_name) in zip(confusion_matrices, models_run):
    print(model_name)
    f = open(f"./confusion_matrix_{model_iteration}.npy", "wb")
    np.save(f, conf_mat)
    f.close()
    conf_mat = np.mean(conf_mat, axis=2)
    print("\nConfusion Matrix:")    
    for row in conf_mat:
        print(str(list(np.round(row).astype(int))).replace("\n", "").replace(",", ""))

    TP = np.diag(conf_mat)
    FP = conf_mat.sum(axis=0) - TP
    FN = conf_mat.sum(axis=1) - TP
    TN = conf_mat.sum() - (FP + FN + TP)

    accuracy = np.round((TP+TN)/(TP+FP+FN+TN), 3)
    precision = np.round(TP/(TP+FP), 3)
    recall = np.round(TP/(TP+FN), 3)
    f1 = np.round(2 * precision*recall/(precision+recall), 3)

    file = open(f"Evaluation_{model_iteration}.txt", "w")
    s = "Name".ljust(15) + " | " + "Accuracy".ljust(9) + " | " + "Precision".ljust(9) + " | " + "Recall".ljust(9) + " | " + "f1-Score".ljust(9) +  " |"
    print(s)
    file.write(s + "\n")
    for c, a, p, r, f in zip(categories, accuracy, precision, recall, f1):
        s = str(c).ljust(15) + " | " + str(a).ljust(9) + " | " + str(p).ljust(9) + " | " + str(r).ljust(9) + " | " + str(f).ljust(9) + " |"
        print(s)
        file.write(s  + "\n")

    file.close()

    model_iteration += 1