import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys

# ========== 数据加载与预处理 ==========
# 假设你有 preprocess.py 并且数据路径、参数如下
try:
    import preprocess
except ImportError:
    print("请确保 preprocess.py 在同目录下，并包含 prepro 函数")
    sys.exit(1)

# 数据参数
num_classes = 10
length = 2048
number = 1000
normal = True
rate = [0.7, 0.2, 0.1]
path = r'E:\machinelearning\code\wdcnn_bearning_fault_diagnosis-master\wdcnn_bearning_fault_diagnosis-master\data\1HP'

# 数据预处理
x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(
    d_path=path, length=length,
    number=number, normal=normal,
    rate=rate, enc=True, enc_step=28
)
y_train = np.argmax(y_train, axis=1)
y_valid = np.argmax(y_valid, axis=1)
y_test = np.argmax(y_test, axis=1)

# 转为Tensor并展平
x_train = torch.tensor(x_train, dtype=torch.float32).view(-1, length)
y_train = torch.tensor(y_train, dtype=torch.long)
x_valid = torch.tensor(x_valid, dtype=torch.float32).view(-1, length)
y_valid = torch.tensor(y_valid, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float32).view(-1, length)
y_test = torch.tensor(y_test, dtype=torch.long)

# ========== DataLoader ==========
batch_size = 256
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataset = TensorDataset(x_valid, y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
#在 PyTorch 中，DataLoader 是一个用于批量加载数据的迭代器。它常用于深度学习训练流程，能够自动地将数据集分成小批量（batch），并支持多线程加载、打乱数据等功能。
#通过将数据集传递给 DataLoader，我们可以方便地按批次获取数据，简化训练过程。
#train_loader 是一个可迭代对象（iterator），每次迭代返回一个 batch 的数据和标签，通常是元组 (inputs, targets)

learning_rate = 0.01
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 配置网络结构，包含输入层、隐藏层、输出层大小
#layer_sizes = [length, 2048,1024, 512, 256, 128, 64, 32, num_classes] #83.7% # 可根据需要修改，例如 [输入, 隐层1, 隐层2, ..., 输出]
layer_sizes = [length, 50000,1000, num_classes]  #  88.60%
layer_sizes = [length, 500000, num_classes]  #  81.10%

# 手动初始化参数
weights = []
biases = []#创建两个空列表，用于存储每一层的权重矩阵（weights）和偏置向量（biases）。
for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):#layer_sizes[:-1]获取从列表开头到倒数第二个元素的所有元素layer_sizes[1:]：获取从第二个元素到列表结尾的所有元素
    #遍历神经网络中所有相邻的层对
    #自动获取每对层的输入和输出尺寸
    #为每个连接初始化权重矩阵和偏置向量
    #zip函数将多个可迭代对象（这里是两个列表）中对应位置的元素打包成元组，然后返回由这些元组组成的迭代器。举例:
    # layer_sizes = [784, 256, 128, 10]
    #layer_sizes[:-1] = [784, 256, 128]
    #layer_sizes[1:]  = [256, 128, 10]
    #zip(layer_sizes[:-1], layer_sizes[1:]) → [(784, 256), (256, 128), (128, 10)]    
    W = torch.randn(in_size, out_size, device=device) * torch.sqrt(torch.tensor(2 / in_size))#生成形状为 (in_size, out_size) 的矩阵,均值为0的正态分布
    #权重初始化的科学依据：
    # 方差守恒：使每层输出的方差 ≈ 输入的方差
    b = torch.zeros(out_size, device=device)#创建长度为 out_size 的向量,所有元素初始化为 0.0
    weights.append(W)
    biases.append(b)

# 激活函数及其导数
def relu(x):
    return torch.clamp(x, min=0)

def relu_grad(x):
    return (x > 0).float()

# Softmax + 交叉熵损失
def softmax(x):
    x_exp = torch.exp(x - x.max(dim=1, keepdim=True).values)
    return x_exp / x_exp.sum(dim=1, keepdim=True)

def cross_entropy(pred, labels):#该函数用于计算交叉熵损失 pred prediction”（预测）的缩写表示模型对输入数据的预测结果
    N = pred.shape[0]#pred 是一个形状为 [batch_size, num_classes] 的张量，所以 N 就是本批次中样本的个数。
    one_hot = torch.zeros_like(pred)#one-hot 编码（独热编码）是一种常用的分类数据表示方法，主要用于将类别标签转换为机器学习模型可以处理的数值向量。
    one_hot[torch.arange(N), labels] = 1  # 生成one-hot编码 用1在向量中的位置表示标签
    loss = - (one_hot * torch.log(pred + 1e-8)).sum() / N  # 计算平均loss，这里加上一个很小的数1e-8，是为了防止出现log(0)时出现负无穷大的情况。
    return loss, one_hot

# 训练循环
# 这个循环是神经网络的训练主循环，每次遍历一个 epoch（轮），对所有训练数据进行前向传播、损失计算、反向传播和参数更新。具体步骤如下：
for epoch in range(num_epochs):#外层循环：控制训练的总轮数，每轮都要遍历全部训练数据。
    total_loss = 0
    for batch_x, batch_y in train_loader:#遍历训练集
        x = batch_x.to(device)
        y = batch_y.to(device)
        N = x.shape[0]#x 是一个形状为 [batch_size, 784] 的张量（每行为一张图片的特征），所以 N 就是当前批次中图片的数量。

        # 前向传播
        activations = [x]#用途：初始化一个列表，保存每一层的激活值（即每层的输出）。
        #含义：x 是当前 batch 的输入数据（形状为 [batch_size, 784]），作为输入层的激活值。
        pre_acts = []#后续每一层的线性变换结果（即 $z = a_{l-1}W + b$）会被存入这里
        for W, b in zip(weights[:-1], biases[:-1]):#遍历所有隐藏层的权重和偏置（不包括输出层）。
            z = activations[-1] @ W + b#计算当前层的线性输出
            #@ 是矩阵乘法运算符（等价于 torch.matmul）
            pre_acts.append(z)#将当前层的线性输出 $z$ 保存到 pre_acts 列表中，便于后续反向传播时使用。
            a = relu(z)#对线性输出 $z$ 应用 ReLU 激活函数，得到当前层的激活值
            activations.append(a)#当前层的激活值 $a$ 保存到 activations 列表中
        
        # 输出层
        z_out = activations[-1] @ weights[-1] + biases[-1]#计算输出层的线性输出
        #@ 是矩阵乘法，activations[-1] 是最后一层隐藏层的输出，weights[-1] 和 biases[-1] 是输出层的权重和偏置。
        pre_acts.append(z_out)#保存输出层的线性输出，便于反向传播时使用
        y_pred = softmax(z_out)#对输出层的线性输出应用 softmax，得到每个类别的概率预测。

        # 损失
        loss, one_hot = cross_entropy(y_pred, y)#计算当前 batch 的交叉熵损失和 one-hot 编码标签。
        total_loss += loss.item()#累计损失，用于后续计算平均损失。用法：loss.item() 将张量转换为 Python 数值。

        # 反向传播
        grads_W = [None] * len(weights)#创建一个与 weights 列表长度相同的列表，用于存储每层的梯度。
        grads_b = [None] * len(biases)

        # 输出层梯度
        dL_dz = (y_pred - one_hot) / N  # [N, output]#计算输出层的梯度  y_pred = softmax(z_out)得到每个类别的概率预测。
        grads_W[-1] = activations[-1].t() @ dL_dz#计算输出层权重的梯度。
        #.t() 是转置，@ 是矩阵乘法
        grads_b[-1] = dL_dz.sum(dim=0)#计算输出层偏置的梯度

        # 隐层梯度
        for i in range(len(weights)-2, -1, -1):#反向遍历所有隐藏层，计算每层的梯度。range(start, stop, step)，这里从倒数第二层到第一层。
            dL_dz = dL_dz @ weights[i+1].t() * relu_grad(pre_acts[i])#链式法则，先传播梯度，再乘以激活函数的导数（ReLU）
            grads_W[i] = activations[i].t() @ dL_dz#计算当前层权重的梯度
            grads_b[i] = dL_dz.sum(dim=0)#计算当前层偏置的梯度。

        # 更新参数
        with torch.no_grad():
            for i in range(len(weights)):#遍历所有层，更新权重和偏置。
                weights[i] -= learning_rate * grads_W[i]#用梯度下降法更新权重。
                biases[i]  -= learning_rate * grads_b[i]#用梯度下降法更新偏置。

    avg_loss = total_loss / len(train_loader)#计算平均损失
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")#打印当前轮次的损失


# 验证集准确率
with torch.no_grad():
    correct = 0#统计预测正确的样本数
    total = 0#统计总样本数
    for batch_x, batch_y in valid_loader:#遍历验证集，每次取出一个 batch 的数据和标签。
        x = batch_x.to(device)
        y = batch_y.to(device)
        a = x#初始化输入，作为第一层的激活值。
        for W, b in zip(weights[:-1], biases[:-1]):#遍历所有隐藏层，做前向传播。zip 配对权重和偏置。
            a = relu(a @ W + b)#每一层做线性变换加 ReLU 激活
        logits = a @ weights[-1] + biases[-1]#计算输出层的线性输出
        preds = logits.argmax(dim=1)#取每个样本预测概率最大的类别作为预测结果。.argmax(dim=1) 沿类别维度取最大值索引。
        correct += (preds == y).sum().item()#统计本 batch 预测正确的样本数并累加。
        #用法：(preds == y) 得到布尔张量，.sum() 统计 True 的个数，.item() 转为 Python 数值。
        total += y.size(0)#y.size(0) 得到 batch 大小，累加本 batch 的样本数。
    print(f"Validation Accuracy: {correct/total*100:.2f}%")

# 测试集准确率
with torch.no_grad():
    correct = 0#统计预测正确的样本数
    total = 0#统计总样本数
    for batch_x, batch_y in test_loader:#遍历测试集，每次取出一个 batch 的数据和标签。
        x = batch_x.to(device)
        y = batch_y.to(device)
        a = x#初始化输入，作为第一层的激活值。
        for W, b in zip(weights[:-1], biases[:-1]):#遍历所有隐藏层，做前向传播。zip 配对权重和偏置。
            a = relu(a @ W + b)#每一层做线性变换加 ReLU 激活
        logits = a @ weights[-1] + biases[-1]#计算输出层的线性输出
        preds = logits.argmax(dim=1)#取每个样本预测概率最大的类别作为预测结果。.argmax(dim=1) 沿类别维度取最大值索引。
        correct += (preds == y).sum().item()#统计本 batch 预测正确的样本数并累加。
        #用法：(preds == y) 得到布尔张量，.sum() 统计 True 的个数，.item() 转为 Python 数值。
        total += y.size(0)#y.size(0) 得到 batch 大小，累加本 batch 的样本数。
    print(f"Test Accuracy: {correct/total*100:.2f}%")