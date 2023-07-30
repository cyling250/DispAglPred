from torch import nn


class NN(nn.Module):
    def __init__(self, input_size, output_size,hidden_size):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)  # input_layer
        self.x1 = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.Tanh()  # 要让模型能够非线性拟合，需要增加激活层
        )
        self.output = nn.Linear(hidden_size, output_size)  # output_layer

    def forward(self, x):
        x = self.input(x)
        x = self.x1(x)
        return self.output(x)
