import torchvision
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, input_size=128, embedding_size=64, pretrained=False):
        nn.Module.__init__(self)
        self.model = torchvision.models.alexnet(pretrained=pretrained)
        self.model.classifier = nn.Linear(in_features=9216, out_features=embedding_size, bias=True)

    def forward(self, x):
        return self.model(x)