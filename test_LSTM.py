from make_data import *
import xlwt
from LSTM import *

# 判断GPU加速是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
if device == torch.device("cuda:0"):
    print("GPU加速已启用")

# 数据集制作
PW, SW, step = read_file_x("quake_data")
X, Y = read_cjwyj("弹性35.xlsx")  # 读取层间位移角数据

for i in range(len(step)):
    print("\r正在进行步长归一化:", i, end="")  # 步长归一化
    PW[i] = freq_conversion(PW[i], step[i], 0.02)
    SW[i] = freq_conversion(SW[i], step[i], 0.02)
    if len(PW[i]) < 2000:  # 长度规整到2000
        PW[i].extend([0 for i in range(2000 - len(PW[i]))])
        SW[i].extend([0 for i in range(2000 - len(SW[i]))])
    else:
        PW[i] = PW[i][0:2000]
        SW[i] = SW[i][0:2000]
    if len(X[i]) < 2000:
        X[i].extend([0 for i in range(2000 - len(X[i]))])
    else:
        X[i] = X[i][0:2000]
# 划分训练集与验证集，选取第10，20，30，40，50，60，70，80，90，100条地震波作为测试集
PW = MinMaxScaler(PW, 35)
SW = MinMaxScaler(SW, 35)  # 归一化到小震情况下
X = MinMaxScaler(X, 1)
test_x = torch.Tensor([PW[97]])
test_x = test_x.unsqueeze(-1)
test_y = torch.Tensor([X[97]])

# 加载模型
model = LSTM_net().to(device)
if os.path.exists("LSTM_model.pt"):
    model.load_state_dict(torch.load("LSTM_model.pt"))  # 如果模型存在，则加载
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
xls.save("LSTM_预测数据.xls")

