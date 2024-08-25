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
import albumentations as A
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from collections import defaultdict
import gc

# self defined functions
from fusion_model_v2 import FusionModel, EarlyFusionModel, MetadataModel
from Conventional_models import *

# Dataset
from NINA_dataset_v2 import NINADataset
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

def confusion_matrix(model, dataloader, class_names, device):
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

def create_model(model_type, num_meta_features, output_size):
    if model_type == "ResNet18":
        return ResNet18Model(num_meta_features, output_size)
    elif model_type == "AlexNet":
        return AlexNetModel(num_meta_features, output_size)
    elif model_type == "Inception":
        return InceptionModel(num_meta_features, output_size)
    elif model_type == "ResNet50":
        return ResNet50Model(num_meta_features, output_size)
    elif model_type == "EfficientNet":
        return EfficientNetModel(num_meta_features, output_size)
    elif model_type == "EfficientNet1":
        return EfficientNet2Model(num_meta_features, output_size)
    elif model_type == "EfficientNet2":
        return EfficientNet3Model(num_meta_features, output_size)
    elif model_type == "EfficientNet3":
        return EfficientNet4Model(num_meta_features, output_size)
    elif model_type == "EfficientNet4":
        return EfficientNet5Model(num_meta_features, output_size)
    elif model_type == "EfficientNet5":
        return EfficientNet6Model(num_meta_features, output_size)
    elif model_type == "EfficientNet6":
        return EfficientNet1Model(num_meta_features, output_size)
    elif model_type == "EfficientNet7":
        return EfficientNet7Model(num_meta_features, output_size)
    elif model_type == "ViT":
        return ViTModel(num_meta_features, output_size)
    elif model_type == "MetadataModel":
        return MetadataModel(num_meta_features, output_size)
    elif model_type == "Fusion":
        return FusionModel(num_meta_features, output_size)
    elif model_type == "EarlyFusion":
        return EarlyFusionModel(num_meta_features, output_size)
    elif model_type == "MetadataModel":
        return MetadataModel(num_meta_features, output_size)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, model_iteration, num_epochs=25, num_finished=0):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1e10
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        if(epoch < num_finished):
            print("Skipping")
            scheduler.step()
            continue
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            for inputs, metadata, targets in dataloaders[phase]:
                inputs = inputs.to(device)
                metadata = metadata.to(device)
                targets = targets.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, metadata)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, targets)
                    
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * targets.size(0)
                running_corrects += torch.sum(preds == targets.data)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{}; Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
            
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

def create_weighted_sampler(annotations, class_names):
    class_counts = {class_name: 0 for class_name in class_names}
    for ann in annotations:
        class_counts[ann['Species']] += 1

    weights = []
    for ann in annotations:
        weights.append(1.0 / class_counts[ann['Species']])
    
    return WeightedRandomSampler(weights, len(annotations))

def get_samples(bp):
    with open(bp + "metadata.json") as f:
        metadata = json.load(f)

    categories = metadata['categories']
    annotations = metadata['annotations']


    return categories, annotations

def define_transforms():
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])
    transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((256, 256)), # only if 512x512 is too large for GPU
            transforms.Normalize(mean, std)
        ])

    albu_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=45, p=1.0),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=1.0),
        A.augmentations.dropout.coarse_dropout.CoarseDropout (max_height=32, max_width=32, min_height=None, min_width=None)
    ])

    return transform, albu_transform

def model_evaluation(matrices, models_run, categories):
    model_iteration = 0 # used to save each model individually
    for (conf_mat, model_name) in zip(matrices, models_run):
        print(model_name)
        f = open(f"./confusion_matrix_{model_iteration}.npy", "wb")
        np.save(f, conf_mat)
        f.close()
        
        print("\nConfusion Matrix:")    
        for row in conf_mat:
            print(str(list(np.round(row).astype(int))).replace("\n", "").replace(",", ""))
        print("-"*80)
        TP = np.diag(conf_mat)
        FP = conf_mat.sum(axis=0) - TP
        FN = conf_mat.sum(axis=1) - TP
        TN = conf_mat.sum() - (FP + FN + TP)

        accuracy = np.round((TP+TN)/(TP+FP+FN+TN), 3)
        precision = np.round(TP/(TP+FP), 3)
        recall = np.round(TP/(TP+FN), 3)
        f1 = np.round(2 * precision*recall/(precision+recall), 3)
        kappa = np.round(cohen_kappa_score(conf_mat), 3)

        file = open(f"Evaluation_{model_iteration}.txt", "w")
        s = "Name".ljust(15) + " | " + "Accuracy".ljust(9) + " | " + "Precision".ljust(9) + " | " + "Recall".ljust(9) + " | " + "f1-Score".ljust(9) +  " | " + "Kappa".ljust(9) + " |"
        print(s)
        file.write(s + "\n")
        for c, a, p, r, f in zip(categories, accuracy, precision, recall, f1):
            s = str(c).ljust(15) + " | " + str(a).ljust(9) + " | " + str(p).ljust(9) + " | " + str(r).ljust(9) + " | " + str(f).ljust(9) + " | " + str(kappa).ljust(9) + " |"
            print(s)
            file.write(s  + "\n")

        file.close()
        print("="*80, "\n")
        model_iteration += 1

def remove_classes(annotations, c):
    new_annotations = []
    for anot in annotations:
        if (anot['Species'] not in c):
            new_annotations.append(anot)
    
    return new_annotations

def keep_classes(annotations, c):
    new_annotations = []
    for anot in annotations:
        if (anot['Species'] in c):
            new_annotations.append(anot)
    
    return new_annotations

def combine_change_class(annotations, src, dst):
    print("Replacing:", src, "with:", dst)
    new_anot = []
    for anot in annotations:
        if(anot['Species'] == src):
            anot['Species'] = dst
        new_anot.append(anot)
    return new_anot

def cohen_kappa_score(conf_matrix):
    n_classes = conf_matrix.shape[0]
    
    # Calculate observed accuracy (Po)
    Po = np.trace(conf_matrix) / conf_matrix.sum()
    
    # Calculate expected accuracy (Pe)
    Pe = sum((conf_matrix.sum(axis=0)[i] * conf_matrix.sum(axis=1)[i]) for i in range(n_classes)) / (conf_matrix.sum() ** 2)
    
    # Calculate Cohen's Kappa score (k)
    k = (Po - Pe) / (1 - Pe)
    
    return k

def main():
    # Data loading
    bp = "/cluster/home/aslakto/ondemand/data/viltkamera/NINA/Images/"
    categories, annotations = get_samples(bp)

    # some class combination to reduce classes
    annotations = combine_change_class(annotations, "Squirrel", "Rodent")
    annotations = combine_change_class(annotations, "Rabbit", "Rodent")
    # for anot in annotations:
        # if(anot['Species'] != "Deer"):
            # anot['Species'] = "Not Deer"
    categories = {}
    for anot in annotations:
        c = anot['Species']
        if c not in categories.keys():
            categories[c] = 1
        else:
            categories[c] += 1
    class_names = [cat for cat in categories.keys()]
    for anot in annotations:
        anot['Species_ID'] = class_names.index(anot['Species'])

    # transforms used
    transform, albu_transform = define_transforms()
    
    # reduce size of whole dataset for trial runs (range 0-1, lower number means less data)
    # annotations = reduce_sample_size(annotations, 'Species_ID', 0.005)

    # model configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DL_models = ["EfficientNet3"]    
    losses = [nn.CrossEntropyLoss()]
    confusion_matrices = []
    models_run = []
    model_iteration = 3 # used to save each model individually

    # data splitting and dataloader creation
    annot_train, annot_test= train_test_split(annotations, test_size=0.1)
    annot_train, annot_val = train_test_split(annot_train, test_size=0.1)

    dataset_train = NINADataset(annot_train, bp, transform, augment_transform=albu_transform, augment=True)
    dataset_val = NINADataset(annot_val, bp, transform)
    dataset_test = NINADataset(annot_test, bp, transform)
    
    wrs = create_weighted_sampler(annot_train, list(categories.keys()))

    bz = 12
    dataloaders = {
        'train': DataLoader(dataset_train, batch_size=bz, sampler=wrs, num_workers=8), # training dataset augmented
        'val': DataLoader(dataset_val, batch_size=bz, num_workers=8),
        'test': DataLoader(dataset_test, batch_size=bz, num_workers=8)
    }

    # get length of metadata vector
    img, meta_features, _ = dataset_train[15]
    print(f"Device: {device}")
    print(len(class_names), categories)
    print(f"Full size: {len(annotations)}, Train size: {len(dataset_train)}, validation size: {len(dataset_val)}, test size: {len(dataset_test)}")
    for model_string in DL_models:
        print(model_string)
        for criterion in losses:
            model = create_model(model_string, len(meta_features), len(class_names))
            model = model.to(device)
    
            optimizer_conv = optim.Adam(model.parameters(), lr=1e-4)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
            model = train_model(model, criterion, optimizer_conv, exp_lr_scheduler, dataloaders, device, model_iteration, num_epochs=25)
            conf_mat = confusion_matrix(model, dataloaders['test'], class_names, device)
            confusion_matrices.append(conf_mat)
            models_run.append(model_string)
            
            # Delete model and optimizer
            del model, optimizer_conv
            
            # Call garbage collector
            gc.collect()
            
            # Empty cuda cache
            torch.cuda.empty_cache()
            model_iteration += 1
    
    model_evaluation(confusion_matrices, models_run, class_names)

    

if __name__ == "__main__":
    main()
