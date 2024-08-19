import torch
from torch.hub import load_state_dict_from_url
import torch.nn as nn
import torchvision
from torchvision import models

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
def load_model(code_length):
    model = Resnet50(code_length)
    state_dict = load_state_dict_from_url(model_urls['resnet50'])
    model.load_state_dict(state_dict, strict=False)
    return model  
  
class Resnet50(nn.Module):
    def __init__(self, code_length):
        super(Resnet50, self).__init__()
        basemodel = torchvision.models.resnet50(pretrained=True)
        self.features = nn.Sequential(
            basemodel.conv1,
            basemodel.bn1,
            basemodel.relu,
            basemodel.maxpool,
            basemodel.layer1,
            basemodel.layer2,
            basemodel.layer3,
            basemodel.layer4,
            basemodel.avgpool,
        )
        self.hash_layer = nn.Sequential(
            nn.Linear(2048, code_length),
            nn.Tanh(),
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.hash_layer(x)
        return x

if __name__ == '__main__':
    model1 = load_model(16)
    model1.hash_layer = nn.Identity()
    model2 = torchvision.models.resnet50(pretrained=True)
    model2.fc = nn.Identity()

    x = torch.randn((1, 3, 224, 224))
    y1 = model1(x)
    y2 = model2(x)
    print()