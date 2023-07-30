import torch
from torch import nn


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            # 输入大小[batch_Size,1,20,100]
            torch.nn.Conv2d(in_channels=1,
                            out_channels=4,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4,
                            out_channels=8,
                            kernel_size=3,
                            stride=1,
                            padding=1),  # [4,8,12,1]
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=8,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=1),  # [4,8,12,1]
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16,
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            padding=1),  # [4,8,12,1]
            torch.nn.Tanh(),
        )
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(in_features=768, out_features=21)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # 保留batch，将后面的乘到一起
        x = self.output(x)
        return x


if __name__ == "__main__":
    x = CNN()
    data = torch.rand(1, 1, 20, 100)
    out = x(data)
