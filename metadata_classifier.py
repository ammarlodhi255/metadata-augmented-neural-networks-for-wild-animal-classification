# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
import numpy as np
# from torchvision import transforms
import time
import copy
import json
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from collections import defaultdict
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter
from itertools import combinations, product
from shutil import copyfile
import os
# self defined functions
from fusion_model_v2 import FusionModel, EarlyFusionModel, MetadataModel


def get_samples(bp):
    with open(bp + "metadata.json") as f:
        metadata = json.load(f)

    categories = metadata['categories']
    annotations = metadata['annotations']


    return categories, annotations

def confusion_matrix(model, dataloader, n_classes, device):
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    with torch.no_grad():
        for metadata, labels in dataloader:
            labels = labels.to(device).to(torch.float32)
            metadata = metadata.to(device)
            outputs = model(metadata, metadata)
            _, preds = torch.max(outputs, 1)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                try:
                    conf_matrix[t.long(), p.long()] += 1
                except:
                    print(t.long(), p.long())

    return conf_matrix

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, model_iteration, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1e10
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            for metadata, targets in dataloaders[phase]:
                metadata = metadata.to(device)
                targets = targets.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(metadata, metadata)
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
            epoch_acc = running_corrects / dataset_sizes[phase] * 100

            print('{}; Loss: {:.4f} Acc: {:.4f}, corrects: {:d}/{:d}'.format(
                phase, epoch_loss, epoch_acc, running_corrects, dataset_sizes[phase]))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f"./Metadata_only/weights_{model_iteration}.bin")

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_accuracy_history.append(epoch_acc)
            if phase == 'val':
                valid_loss_history.append(epoch_loss)
                valid_accuracy_history.append(epoch_acc)
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def prf(conf_mat):
    n_classes = conf_mat.shape[0]
    TPS = conf_mat.diagonal()
    FPS = np.array([np.sum(conf_mat[:, i]) - TPS[i] for i in range(n_classes)])
    FNS = np.array([np.sum(conf_mat[i, :]) - TPS[i] for i in range(n_classes)])
    F1S = 2*TPS/(2*TPS + FPS + FNS)
    
    return TPS, FPS, FNS, F1S

def cohen_kappa_score(conf_matrix):
    n_classes = conf_matrix.shape[0]
    
    # Calculate observed accuracy (Po)
    Po = np.trace(conf_matrix) / conf_matrix.sum()
    
    # Calculate expected accuracy (Pe)
    Pe = sum((conf_matrix.sum(axis=0)[i] * conf_matrix.sum(axis=1)[i]) for i in range(n_classes)) / (conf_matrix.sum() ** 2)
    
    # Calculate Cohen's Kappa score (k)
    k = (Po - Pe) / (1 - Pe)
    
    return k


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
    for anot in annotations:
        if(anot['Species'] == src):
            anot['Species'] = dst
    
    return annotations


def ret_vector(annotations, values, class_names):
    X = []
    y = []
    temp = ["Datetime", "Places", "Position & Temperature", "Scene features"]
    included_data = [temp[i] for i in range(len(temp)) if values[i+1] == 1]
    s = f"Batch size: {values[0]}, Data: {included_data}, Categories: {class_names}"
    for anot in annotations:
        temp = []
        if(values[1]):
            temp.extend(anot['datetime_vector']) 
        if(values[2]):
            temp.extend(anot['env_vector'][1:366])
        if(values[3]):
            if(anot['Temperature'] is None):
                temp.append(np.nan)
            else:
                t = str(anot['Temperature'])
                t = t.replace("C", "")
                if(len(t) < 1):
                    temp.append(np.nan)
                else:
                    temp.append(float(t))
            temp.extend([float(anot['Latitude']), float(anot['Longitude'])])
        if(values[4]):
            temp.extend(anot['env_vector'][366:])
            
        X.append(temp)
        y.append(anot['Species_ID'])

    X = np.array(X).astype(np.float32)

    # replace nans by the average of the row (maybe not smart)
    if(np.isnan(np.sum(X))):
        row_means = np.nanmean(X, axis=1)
        for i in range(X.shape[0]):
            X[i, np.isnan(X[i, :])] = row_means[i]

    y = np.array(y)
    return X, y, s

train_loss_history = []
valid_loss_history = []
train_accuracy_history = []
valid_accuracy_history = []
def main():
    global train_loss_history
    global valid_loss_history
    global train_accuracy_history
    global valid_accuracy_history
    # Data loading
    bp = "/media/user-1/CameraTraps/NINA/Images/"
    categories, annotations = get_samples(bp)

    # combine/remove classes
    annotations = combine_change_class(annotations, "Squirrel", "Rodent")
    annotations = combine_change_class(annotations, "Rabbit", "Rodent")
    annotations = remove_classes(annotations, ['Cattle', 'Boar', 'Bear'])
    rest_class = ['Lynx', 'Bird', 'Wolf', 'Fox', 'Deer', 'Weasel', 'Cat', 'Sheep', 'Rodent']

    comb = []

    # Iterate through different lengths of combinations (pairs, triplets, etc.)
    for i in range(2, len(rest_class) + 1):
        # Generate combinations of length i
        c = combinations(rest_class, i)

        # Add all combinations to comb list
        comb.extend(c)
    
    bzs = [64]
    datetime = [0, 1]
    places = [0, 1]
    scene_features = [0, 1]
    pos_temp = [0, 1]


    model_configurations = list(product(bzs, datetime, places, pos_temp, scene_features))
    model_configurations = [config for config in model_configurations if not all(val == 0 for val in config[1:])]
    files = os.listdir("./Metadata_only_old")
    print(f"Total combinations: {len(comb)*len(model_configurations)}")
    itr = 0
    for c in comb:
        kc = []
        kc.extend(c)
        new_annotations = keep_classes(annotations, kc)
        new_categories = {}
        for anot in new_annotations:
            c = anot['Species']
            if c not in new_categories.keys():
                new_categories[c] = 1
            else:
                new_categories[c] += 1
        categories = new_categories
        class_names = list(categories.keys())
        for anot in new_annotations:
            anot['Species_ID'] = class_names.index(anot['Species'])
        for config in model_configurations:
            X, y, s = ret_vector(new_annotations, config, class_names)
            if((config[2] == 0) and (config[4] == 0) and (f"error_{itr}" not in files)):
                print("="*80)
                print("Coyping:", s)
                copyfile(f"./Metadata_only_old/confusion_matrix_{itr}.npy", f"./Metadata_only/confusion_matrix_{itr}.npy")
                copyfile(f"./Metadata_only_old/evaluation_{itr}.txt", f"./Metadata_only/evaluation_{itr}.txt")
                copyfile(f"./Metadata_only_old/weights_{itr}.bin", f"./Metadata_only/weights_{itr}.bin")
            else:
                print("="*80)
                print("\n")
                print(s)
                print(X.shape)
                # data splitting and dataloader creation
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

                oversample = BorderlineSMOTE()
                X_train, y_train = oversample.fit_resample(X_train, y_train)

                dataset_train = data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
                dataset_val = data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
                dataset_test = data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

                dataloaders = {
                    'train': DataLoader(dataset_train, batch_size=config[0], num_workers=8),
                    'val': DataLoader(dataset_val, batch_size=config[0], num_workers=8),
                    'test': DataLoader(dataset_test, batch_size=config[0], num_workers=8)
                }
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                sample_x, sample_y = dataset_train[0]
                num_classes = len(Counter(y).keys())
                model = MetadataModel(len(sample_x), num_classes)
                model = model.to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer_conv = optim.Adam(model.parameters(), lr=1e-3)
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


                try: 
                    model = train_model(model, criterion, optimizer_conv, exp_lr_scheduler, dataloaders, device, itr, num_epochs=25)
                    conf_mat = confusion_matrix(model, dataloaders['test'], num_classes, device)
                    kappa = cohen_kappa_score(conf_mat)
                    TPS, FPS, FNS, F1S = prf(conf_mat)

                    total_FP = FPS.sum()
                    total_FN = FNS.sum()
                    total = conf_mat.sum()
                    total_TP = TPS.sum()
                    FPR = total_FP / (total + total_TP)
                    FNR = total_FN / (total + total_TP)
                    F1_score = 2*total_TP/(2*total_TP + total_FP + total_FN)

                    f = open(f"./Metadata_only/evaluation_{itr}.txt", "w")
                    f.write(s + "\n")
                    print("\nConfusion Matrix:")    
                    for row in conf_mat:
                        s = str(list(np.round(row).astype(int))).replace("\n", "").replace(",", "")
                        print(s)
                        f.write(s + "\n")

                    print("totals:\nACC \t FPR \t FNR \t kappa")
                    print(round(total_TP/total, 3), "\t", round(FPR, 3), "\t", round(FNR, 3), "\t", round(kappa, 3))
                    f.write("totals:\nACC \t FPR \t FNR \t kappa\n")
                    f.write(str(round(total_TP/total, 3)) + "\t" + str(round(FPR, 3)) + "\t" + str(round(FNR, 3)) +"\t" + str(round(kappa, 3)))
                    f.close()
                    np.save(f"./Metadata_only/confusion_matrix_{itr}.npy", conf_mat)
                    history = torch.tensor([train_loss_history, train_accuracy_history, valid_loss_history, valid_accuracy_history])
                    history = history.numpy()
                    train_loss_history = []
                    valid_loss_history = []
                    train_accuracy_history = []
                    valid_accuracy_history = []
                    np.save(f"./Metadata_only/history_{itr}.npy", history)                    
                except:
                    print("Could Not finish task:")
                    print(s, f"itr: {itr}")
                    f = open(f"./Metadata_only/error_{itr}", "w")
                    f.write(s)
                    f.close()
            itr += 1
        

if __name__ == "__main__":
    main()