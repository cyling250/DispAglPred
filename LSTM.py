import torch
from torch import nn


class LSTM_net(nn.Module):
    def __init__(self):
        super(LSTM_net, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,  # 输入通道数
            hidden_size=128,  # 自定义hidden层神经元数量
            num_layers=2,  # lstm隐层长度，隐层长度不能过长
            batch_first=True  # 数据的第二个维度是batch
        )
        self.tanh = nn.Tanh()
        self.l1 = nn.Linear(128, 21)  # l1的作用是将输出的16维隐层神经元转化成一维的

    def forward(self, x):
        x, (ht, ct) = self.lstm(x)
        ht = self.tanh(ht[0])
        ht = self.l1(ht)  # 通过全连接层
        return ht


if __name__ == "__main__":
    data = torch.rand(4, 2000, 1)
    lstm = LSTM_net()
    out = lstm(data)
    print(out.size())
