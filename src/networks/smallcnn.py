from torch import nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """ Following the VGGnet based on VGG16 but for smaller input (64x64)
        Check this blog for some info: https://learningai.io/projects/2017/06/29/tiny-imagenet.html
    """

    def __init__(self, num_classes=1000, last_relu=True):
        super().__init__()

        self.last_relu = last_relu

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.fc6 = nn.Linear(in_features=128 * 4 * 4, out_features=256, bias=True)
        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(in_features=256, out_features=num_classes, bias=True)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

    def forward(self, x):
        h = self.features(x)
        h = h.view(x.size(0), -1)
        if self.last_relu:
            h = F.dropout(F.relu(self.fc6(h)), 0.5)
        else:
            h = F.dropout(self.fc6(h), 0.5)
        h = self.fc(h)
        return h


def smallcnn(num_out=100, pretrained=False):
    if pretrained:
        raise NotImplementedError
    return SmallCNN(num_out)
