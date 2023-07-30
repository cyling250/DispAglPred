from make_data import *
import xlwt
from CNN import *
import torch

# 判断GPU加速是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == torch.device("cuda:0"):
    print("GPU加速已启用")

# 数据集制作
PW, SW, step = read_file_x("quake_data")
X, Y = read_cjwyj("弹性35.xlsx")

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
PW = dim1_to_dim2(PW)
SW = dim1_to_dim2(SW)  # 数据变换到二维

# 读入需要测试的数据
test_x = torch.Tensor([PW[97]])
test_y = torch.Tensor([X[97]])
test_x = test_x.unsqueeze(1)
# 加载模型
model = CNN().to(device)
if os.path.exists("CNN_model.pt"):
    model.load_state_dict(torch.load("CNN_model.pt"))  # 如果模型存在，则加载
if torch.cuda.is_available():
    # 将模型、数据全部导入到cuda上
    model = model.cuda()
    test_x = test_x.to(device)
test_y_pred = model(test_x)  # 计算验证集前向传播
test_y_pred = test_y_pred.tolist()
test_y = test_y.tolist()
xls = xlwt.Workbook()
sheet = xls.add_sheet("1")
for i in range(len(test_y_pred[0])):
    sheet.write(i, 0, str(test_y_pred[0][i]))
    sheet.write(i, 1, str(test_y[0][i]))
xls.save("CNN_预测数据.xls")
