# 库导入部分
from make_data import *
import xlwt
from NN import *
import torch.utils.data as Data
import torch

# 判断GPU加速是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
if device == torch.device("cuda:0"):
    print("GPU加速已启用")

# 数据集制作
PW, SW, step = read_file_x("quake_data")  # 读入地震动信息
X, Y = read_cjwyj("弹性35.xlsx")  # 读取层间位移角数据

# 步长归一化与长度规整
for i in range(len(step)):
    print("\r正在进行步长归一化:", i, end="")  # 步长归一化到50hz
    PW[i] = freq_conversion(PW[i], step[i], 0.02)
    SW[i] = freq_conversion(SW[i], step[i], 0.02)
    if len(PW[i]) < 2000:  # 长度规整到2000
        PW[i].extend([0 for i in range(2000 - len(PW[i]))])
        SW[i].extend([0 for i in range(2000 - len(SW[i]))])
    else:
        PW[i] = PW[i][0:2000]
        SW[i] = SW[i][0:2000]

# 将输入归一化到35cm/s^2，输出归一化到1
PW = MinMaxScaler(PW, 35)
SW = MinMaxScaler(SW, 35)
X = MinMaxScaler(X, 1)

# 划分数据集，将前90个数据分为训练集，将后10个数据分为测试集
# PW:[100,2000],X:[100,2000]
train_x = torch.Tensor(PW[:-10])
train_y = torch.Tensor(X[:-10])
test_x = torch.Tensor(PW[-10:])
test_y = torch.Tensor(X[-10:])
train_dataset = Data.TensorDataset(train_x, train_y)
train_data = Data.DataLoader(
    dataset=train_dataset,
    batch_size=4,
    shuffle=True,
    drop_last=True
)
test_dataset = Data.TensorDataset(test_x, test_y)
test_data = Data.DataLoader(
    dataset=train_dataset,
    batch_size=1,
    shuffle=True,
    drop_last=True
)
# 数据集读取完毕

# 加载模型
train_loss_data = []  # 存储损失函数结果
test_loss_data = []
model = NN(2000, 21, 128).to(device)  # 模型加载到GPU
if os.path.exists("NN_model.pt"):
    model.load_state_dict(torch.load("NN_model.pt"))  # 如果模型存在，则加载
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 定义优化器,学习率
loss_func = nn.MSELoss().to(device)  # 定义损失函数
if torch.cuda.is_available():
    # 将模型、数据全部导入到cuda上
    model = model.cuda()
    loss_func = loss_func.cuda()
train_loss = 0
test_loss = 0
for epoch in range(50):
    # 前向传播计算预测
    for step, (x_batch, y_batch) in enumerate(train_data):
        # 前向计算
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch)
        # 计算并打印损失函数
        train_loss = loss_func(y_pred, y_batch)  # 计算损失函数
        # 梯度逆传播
        optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
        train_loss.backward()  # loss反向传播
        optimizer.step()  # 反向传播后参数更新
    for step, (test_x, test_y) in enumerate(test_data):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        test_y_pred = model(test_x[0])  # 计算验证集前向传播
        test_loss = loss_func(test_y_pred, test_y[0])  # 计算验证集损失函数
    print("epoch:{},train_loss：{:.6f}".format(epoch, train_loss.item()), "test_loss:{:.6f}".format(test_loss.item()))
    train_loss_data.append(train_loss.item())
    test_loss_data.append(test_loss.item())

# 存储loss数据
loss_xls = xlwt.Workbook()
loss_sheet = loss_xls.add_sheet("loss_data")
for i in range(len(train_loss_data)):
    loss_sheet.write(i, 0, train_loss_data[i])
    loss_sheet.write(i, 1, test_loss_data[i])
loss_xls.save("NN_loss数据1.xls")
torch.save(model.state_dict(), "NN_model.pt")
