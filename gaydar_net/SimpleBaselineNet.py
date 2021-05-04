import torch
import torch.nn as nn
import torchvision.models as models

class SimpleBaselineNet(nn.Module):
    """
    A vanilla regression model which takes facial images as input and predicts ones' height
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(*list(models.vgg16(pretrained=True).children())[:-1])
        self.regressor = nn.Sequential(
            # (N, 512, 1, 1)
            nn.MaxPool2d((7, 7)),
            # (N, 512)
            nn.Flatten(),
            # fc1
            #nn.Linear(25088, 4096),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            # fc2
            #nn.Linear(4096, 1024),
            nn.Linear(128, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Dropout(0.4),
            # fc3
            nn.Linear(10, 1),
        )

    def forward(self, x):
        """
        :param x: input image in shape of (N, 3, 224, 224)
        :return out: regression predictions of height (N, )
        """
        x = self.features(x)
        x = self.regressor(x).squeeze()

        return x



