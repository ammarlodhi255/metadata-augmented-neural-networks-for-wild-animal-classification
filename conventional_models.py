import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
from vit_pytorch import ViT
from vit_pytorch.deepvit import DeepViT


class ResNet18Model(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(ResNet18Model, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # Remove the original classifier # type: ignore 

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)
        self.model_name = "ResNet18Model"

    def forward(self, x, metadata):
        x = self.resnet(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.classifier(x)

        return output

class VGG16Model(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(VGG16Model, self).__init__()
        self.vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)  # Load the pretrained VGG16
        self.vgg16.classifier = nn.Identity() # type: ignore 

        self.fc1 = nn.Linear(4096, 128)
        self.fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)
        self.model_name = "VGG16Model"

    def forward(self, x, metadata):
        x = self.vgg16(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.classifier(x)

        return output
    
class AlexNetModel(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(AlexNetModel, self).__init__()
        self.alexnet = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
        self.alexnet.classifier = nn.Identity() # type: ignore 

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, metadata):
        x = self.alexnet(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.classifier(x)

        return output

class InceptionModel(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(InceptionModel, self).__init__()
        self.inception = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Identity()
        self.inception.aux_logits = False

        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, metadata):
        x = self.inception(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.classifier(x)

        return output

class ResNet50Model(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(ResNet50Model, self).__init__()
        self.resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.resnet50.fc = nn.Identity() # type: ignore 

        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, metadata):
        x = self.resnet50(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.classifier(x)

        return output

class EfficientNetModel(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(EfficientNetModel, self).__init__()
        self.efficientnet = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Identity() # type: ignore 

        self.fc1 = nn.Linear(1280, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, metadata):
        x = self.efficientnet(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.classifier(x)

        return output

class EfficientNet1Model(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(EfficientNet1Model, self).__init__()
        self.efficientnet = torchvision.models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Identity()

        self.fc1 = nn.Linear(1280, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, metadata):
        x = self.efficientnet(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.classifier(x)

        return output

class EfficientNet2Model(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(EfficientNet2Model, self).__init__()
        self.efficientnet = torchvision.models.efficientnet_b2(weights=torchvision.models.EfficientNet_B2_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Identity()

        self.fc1 = nn.Linear(1408, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, metadata):
        x = self.efficientnet(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.classifier(x)

        return output

class EfficientNet3Model(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(EfficientNet3Model, self).__init__()
        self.efficientnet = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Identity()

        self.fc1 = nn.Linear(1536, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, metadata):
        x = self.efficientnet(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.classifier(x)

        return output

class EfficientNet4Model(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(EfficientNet4Model, self).__init__()
        self.efficientnet = torchvision.models.efficientnet_b4(weights=torchvision.models.EfficientNet_B4_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Identity()

        self.fc1 = nn.Linear(1792, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, metadata):
        x = self.efficientnet(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.classifier(x)

        return output

class EfficientNet5Model(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(EfficientNet5Model, self).__init__()
        self.efficientnet = torchvision.models.efficientnet_b5(weights=torchvision.models.EfficientNet_B5_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Identity()

        self.fc1 = nn.Linear(2048, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, metadata):
        x = self.efficientnet(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.classifier(x)

        return output

class EfficientNet6Model(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(EfficientNet6Model, self).__init__()
        self.efficientnet = torchvision.models.efficientnet_b6(weights=torchvision.models.EfficientNet_B6_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Identity()

        self.fc1 = nn.Linear(2304, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, metadata):
        x = self.efficientnet(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.classifier(x)

        return output

class EfficientNet7Model(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(EfficientNet7Model, self).__init__()
        self.efficientnet = torchvision.models.efficientnet_b7(weights=torchvision.models.EfficientNet_B7_Weights.DEFAULT)
        self.efficientnet.classifier = nn.Identity() # type: ignore 

        self.fc1 = nn.Linear(2560, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, metadata):
        x = self.efficientnet(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.classifier(x)

        return output
    
class ViTModel(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(ViTModel, self).__init__()

        self.vit = DeepViT(
            image_size = 512,
            patch_size = 32,
            num_classes = num_classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

    def forward(self, x, metadata):
        x = self.vit(x)
        return x