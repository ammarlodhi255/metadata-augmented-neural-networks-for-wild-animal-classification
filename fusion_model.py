import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch

class FusionModel(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(FusionModel, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # Remove the original classifier # type: ignore
        
        self.metadata_fc1 = nn.Linear(num_metadata_features, 64)
        self.metadata_fc2 = nn.Linear(64, 32)
        self.classifier = nn.Linear(512 + 32, num_classes)  # ResNet output + metadata output size
        self.model_name = "Late Fusion Model"

    def forward(self, x, metadata):
        x = self.resnet(x)

        metadata = F.relu(self.metadata_fc1(metadata))
        metadata = F.relu(self.metadata_fc2(metadata))
        fused_features = torch.cat((x, metadata), dim=1)
        output = self.classifier(fused_features)

        return output


class EarlyFusionBlock(nn.Module):
    def __init__(self, block, metadata_size):
        super(EarlyFusionBlock, self).__init__()
        self.block = block
        self.metadata_fc = nn.Linear(metadata_size, block.conv1.out_channels)
        
    def forward(self, x, metadata):
        weight = torch.sigmoid(self.metadata_fc(metadata)).unsqueeze(-1).unsqueeze(-1)
        out = self.block(x)
        out = out * weight
        return out

class EarlyFusionModel(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(EarlyFusionModel, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        
        # Remove the original classifier and replace it with an Identity layer
        self.resnet.fc = nn.Identity()
        
        # Replace each residual block with an EarlyFusionBlock
        for name, module in self.resnet.named_children():
            if name.startswith("layer"):
                early_fusion_blocks = []
                for b in module:
                    early_fusion_blocks.append(EarlyFusionBlock(b, num_metadata_features))
                setattr(self.resnet, name, nn.Sequential(*early_fusion_blocks))

        self.classifier = nn.Linear(512, num_classes)
        self.model_name = "Early Fusion Model"

    def forward(self, x, metadata):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        for name, module in self.resnet.named_children():
            if name.startswith("layer"):
                for block in module:
                    x = block(x, metadata)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.classifier(x)

        return output
    

class MetadataModel(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(MetadataModel, self).__init__()

        self.metadata_fc1 = nn.Linear(num_metadata_features, 128)
        self.metadata_fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)  # ResNet output + metadata output size
        self.model_name = "Metadata Model"

    def forward(self, metadata):
        out = F.relu(self.metadata_fc1(metadata))
        out = F.relu(self.metadata_fc2(out))
        out = self.classifier(out)
        return out
    

class MetadataModel(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(MetadataModel, self).__init__()
        self.metadata_fc1 = nn.Linear(num_metadata_features, 128)
        self.metadata_fc2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)  # ResNet output + metadata output size
        self.model_name = "Metadata Model"

    def forward(self, x, metadata):
        metadata = F.relu(self.metadata_fc1(metadata))
        metadata = F.relu(self.metadata_fc2(metadata))
        output = self.classifier(metadata)
        return output