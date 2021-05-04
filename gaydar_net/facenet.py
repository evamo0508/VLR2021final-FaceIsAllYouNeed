import torch
import torch.nn as nn
import torch.nn.functional as F

from inception_resnet_v1 import InceptionResnetV1

class Model(nn.Module):
    def __init__(self, pretrained='vggface2'):
        super().__init__()
        print("FaceNet pretrained on: {}".format(pretrained))
        self.device = "cpu"
        if pretrained == 'vggface2':
            weights_file = '20180402-114759-vggface2.pt'
        elif pretrained == 'casia-webface':
            weights_file = '20180408-102900-casia-webface.pt'
        resnet = InceptionResnetV1(pretrained=pretrained, weights_file=weights_file, device=self.device)
        self.features = nn.ModuleList(list(resnet.children())[:-1])
        for p in self.features.parameters():
            p.requires_grad = False
            
        self.classifier = nn.Sequential(nn.Linear(512, 128), nn.Linear(128, 1))
        
    def forward(self, x):
        for i, layer in enumerate(self.features):
            if i == 15:
                x = x.view(x.shape[0], -1)
            x = layer(x)
        x = self.classifier(x)
        return x
    
# example
"""
m = Model(pretrained='casia-webface')
X = torch.Tensor(10, 3, 224, 224)
y = m(X)

m = Model(pretrained='vggface2')
X = torch.Tensor(10, 3, 224, 224)
y = m(X)
"""