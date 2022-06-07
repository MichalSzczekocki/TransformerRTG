import torch
from torch import nn
# from torchvision.models import resnet18
import torchvision.models
from torchvision.models.resnet import model_urls


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # model_urls['resnet50'] = model_urls['resnet50'].replace('https://download.pytorch.org/models/', 'https://s3.amazonaws.com/pytorch/models/')
        model_urls['resnet18'] = model_urls['resnet18'].replace('https://download.pytorch.org/models/',
                                                                'https://s3.amazonaws.com/pytorch/models/')

        resnet18 = torchvision.models.resnet18(pretrained=True)
        # resnet50 = torchvision.models.resnet50(pretrained=True)

        self.backbone = resnet18
        # self.backbone = resnet50

        self.fc = nn.Linear(in_features=512, out_features=3)  # 2048 - resnet50  512 - resnet18

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)

        # print(len(x[0])) # 2048 - resnet50  512 - resnet18

        x = x.view(x.size(0), 512)
        x = self.fc(x)

        return x