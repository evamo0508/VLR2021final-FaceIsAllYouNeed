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
            # flatten
            nn.Flatten(),
            # fc1
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # fc2
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            # fc3
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        """
        :param x: input image in shape of (N, 3, 224, 224)
        :return out: regression predictions of height (N, )
        """
        x = self.features(x)
        x = self.regressor(x).squeeze()

        return x



