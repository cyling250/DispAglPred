from make_data import *
import xlwt
from CNN import *
import torch.utils.data as Data

# 判断GPU加速是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == torch.device("cuda:0"):
    print("GPU加速已启用")

# 数据集制作
PW, SW, step = read_file_x("quake_data")  # 读取地震波数据
X, Y = read_cjwyj("弹性35.xlsx")  # 读取层间位移角数据

# 步长与长度处理
for i in range(len(step)):
    print("\r", i, end="")  # 步长归一化到50hz
    PW[i] = freq_conversion(PW[i], step[i], 0.02)
    SW[i] = freq_conversion(SW[i], step[i], 0.02)
    if len(PW[i]) < 2000:  # 长度规整到2000
        PW[i].extend([0 for i in range(2000 - len(PW[i]))])
        SW[i].extend([0 for i in range(2000 - len(SW[i]))])
    else:
        PW[i] = PW[i][0:2000]
        SW[i] = SW[i][0:2000]

# 归一化处理
PW = MinMaxScaler(PW, 35)
SW = MinMaxScaler(SW, 35)  # 归一化到小震情况下
X = MinMaxScaler(X, 1)  # 标签归一化，为了消除量纲的影响，使用MinMaxScaler处理
PW = dim1_to_dim2(PW)
SW = dim1_to_dim2(SW)  # 数据变换到二维

# 数据集划分
train_x = torch.Tensor(PW[:-10])
train_x = train_x.unsqueeze(1)  # 在第二维增加一个维度，表示为图片的深度
train_y = torch.Tensor(X[:-10])
test_x = torch.Tensor(PW[-10:])
test_x = test_x.unsqueeze(1)
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
    dataset=test_dataset,
    batch_size=1,
    shuffle=True,
    drop_last=True
)

# 加载模型
train_loss_data = []  # 存储损失函数结果
test_loss_data = []
model = CNN().to(device)  # 模型加载到GPU

if os.path.exists("CNN_model.pt"):
    model.load_state_dict(torch.load("CNN_model.pt"))  # 如果模型存在，则加载

optimizer = torch.optim.Adam(model.parameters(), lr=10e-8)  # 定义优化器,学习率0.01,L2正则化0.01
loss_func = nn.MSELoss().to(device)  # 定义损失函数

if torch.cuda.is_available():
    # 将模型、数据全部导入到cuda上
    model = model.cuda()
    loss_func = loss_func.cuda()
train_loss = 0
test_loss = 0

for epoch in range(100):
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
    # 选择每隔20个epoch保存一次模型
    for step, (x_batch, y_batch) in enumerate(test_data):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch)
        test_loss = loss_func(y_pred, y_batch)
    print("epoch{}:train_loss：{:.9f},test_loss:{:.9f}".format(epoch, train_loss.item(), test_loss.item()))
    train_loss_data.append(train_loss.item())
    test_loss_data.append(test_loss.item())

# 存储loss数据
loss_xls = xlwt.Workbook()
loss_sheet = loss_xls.add_sheet("1")
for i in range(len(train_loss_data)):
    loss_sheet.write(i, 0, train_loss_data[i])
    loss_sheet.write(i, 1, test_loss_data[i])
loss_xls.save("CNN_loss数据5.xls")
torch.save(model.state_dict(), "CNN_model.pt")

# loss file 1:epoch 100,lr = 0.000001.Save as CNN_model.pt.
