from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Function

# ********************* 二值(+-1) ***********************
# A
class Binary_a(Function):

    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        # *******************ste*********************
        grad_input = grad_output.clone()
        # ****************saturate_ste***************
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


# W
class Binary_w(Function):

    @staticmethod
    def forward(self, input):
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        # *******************ste*********************
        grad_input = grad_output.clone()
        return grad_input


# ********************* 三值(+-1、0) ***********************
class Ternary(Function):

    @staticmethod
    def forward(self, input):
        # **************** channel级 - E(|W|) ****************
        E = torch.mean(torch.abs(input), (3, 2, 1), keepdim=True)
        # **************** 阈值 ****************
        threshold = E * 0.7
        # ************** W —— +-1、0 **************
        output = torch.sign(
            torch.add(torch.sign(torch.add(input, threshold)), torch.sign(torch.add(input, -threshold))))
        return output, threshold

    @staticmethod
    def backward(self, grad_output, grad_threshold):
        # *******************ste*********************
        grad_input = grad_output.clone()
        return grad_input


# ********************* A(特征)量化(二值) ***********************
class activation_bin(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = A
        self.relu = nn.ReLU(inplace=True)

    def binary(self, input):
        output = Binary_a.apply(input)
        return output

    def forward(self, input):
        if self.A == 2:
            output = self.binary(input)
            # ******************** A —— 1、0 *********************
            # a = torch.clamp(a, min=0)
        else:
            output = self.relu(input)
        return output


# ********************* W(模型参数)量化(三/二值) ***********************
def meancenter_clampConvParams(w):
    mean = w.data.mean(1, keepdim=True)
    w.data.sub(mean)  # W中心化(C方向)
    w.data.clamp(-1.0, 1.0)  # W截断
    return w


class weight_tnn_bin(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.W = W

    def binary(self, input):
        output = Binary_w.apply(input)
        return output

    def ternary(self, input):
        output = Ternary.apply(input)
        return output

    def forward(self, input):
        if self.W == 2 or self.W == 3:
            # **************************************** W二值 *****************************************
            if self.W == 2:
                output = meancenter_clampConvParams(input)  # W中心化+截断
                # **************** channel级 - E(|W|) ****************
                E = torch.mean(torch.abs(output), (3, 2, 1), keepdim=True)
                # **************** α(缩放因子) ****************
                alpha = E
                # ************** W —— +-1 **************
                output = self.binary(output)
                # ************** W * α **************
                # output = output * alpha # 若不需要α(缩放因子)，注释掉即可
                # **************************************** W三值 *****************************************
            elif self.W == 3:
                output_fp = input.clone()
                # ************** W —— +-1、0 **************
                output, threshold = self.ternary(input)
                # **************** α(缩放因子) ****************
                output_abs = torch.abs(output_fp)
                mask_le = output_abs.le(threshold)
                mask_gt = output_abs.gt(threshold)
                output_abs[mask_le] = 0
                output_abs_th = output_abs.clone()
                output_abs_th_sum = torch.sum(output_abs_th, (3, 2, 1), keepdim=True)
                mask_gt_sum = torch.sum(mask_gt, (3, 2, 1), keepdim=True).float()
                alpha = output_abs_th_sum / mask_gt_sum  # α(缩放因子)
                # *************** W * α ****************
                output = output * alpha  # 若不需要α(缩放因子)，注释掉即可
        else:
            output = input
        return output


# ********************* 量化卷积（同时量化A/W，并做卷积） ***********************
class Conv2d_Q(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            A=2,
            W=2
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        # 实例化调用A和W量化器
        self.activation_quantizer = activation_bin(A=A)
        self.weight_quantizer = weight_tnn_bin(W=W)

    def forward(self, input):
        # 量化A和W
        bin_input = self.activation_quantizer(input)
        tnn_bin_weight = self.weight_quantizer(self.weight)
        #用-1做padding
        padding_tuple=(self.padding[0],self.padding[0],self.padding[0],self.padding[0])
        bin_input_pad=F.pad(input=bin_input,pad=padding_tuple,mode='constant',value=-1)
        # 用量化后的A和W做卷积
        output = F.conv2d(
            input=bin_input_pad,
            weight=tnn_bin_weight,
            bias=self.bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups)
        return output


# *********************量化(三值、二值)卷积*********************
class Tnn_Bin_Conv2d(nn.Module):
    # 参数：last_relu-尾层卷积输入激活
    def __init__(self, input_channels, output_channels,
                 kernel_size=-1, stride=-1, padding=-1, groups=1, last_relu=0, A=2, W=2):
        super(Tnn_Bin_Conv2d, self).__init__()
        self.A = A
        self.W = W
        self.last_relu = last_relu

        # ********************* 量化(三/二值)卷积 *********************
        self.tnn_bin_conv = Conv2d_Q(input_channels, output_channels,
                                     kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, A=A, W=W)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.tnn_bin_conv(x)
        x = self.bn(x)
        if self.last_relu:
            x = self.relu(x)
        return x


class Net(nn.Module):
    def __init__(self, cfg=None, A=2, W=2):
        super(Net, self).__init__()
        # 模型结构与搭建
        if cfg is None:
            cfg = [16, 32, 64, 10]
        self.tnn_bin = nn.Sequential(
            nn.Conv2d(1, cfg[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(cfg[0]),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Tnn_Bin_Conv2d(cfg[0], cfg[1], kernel_size=5, stride=1, padding=2, A=A, W=W),
            Tnn_Bin_Conv2d(cfg[1], cfg[1], kernel_size=5, stride=1, padding=2, A=A, W=W),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Tnn_Bin_Conv2d(cfg[1], cfg[2], kernel_size=5, stride=1, padding=2, A=A, W=W),
            Tnn_Bin_Conv2d(cfg[2], cfg[3], kernel_size=5, stride=1, padding=2, last_relu=1, A=A, W=W),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.tnn_bin(x)
        x = x.view(x.size(0), -1)
        return x


import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda:0')


# 随机种子——训练结果可复现
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 训练lr调整
def adjust_learning_rate(optimizer, epoch):
    update_list = [10, 20, 30, 40, 50]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.2
    return


# 模型训练
def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # 前向传播
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()  # 求梯度
        optimizer.step()  # 参数更新

        # 显示训练集loss(/100个batch)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item(),
                optimizer.param_groups[0]['lr']))
    return


# 模型测试
def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # 前向传播
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    # 测试准确率
    acc = 100. * float(correct) / len(test_loader.dataset)

    print(acc)


def my_conv(x, w, kernel_size, stride, padding, padding_value):
    batch_size, channel_in, height, width = x.size()
    channel_out, channel_in, kx, ky = w.size()
    x_pad = torch.nn.functional.pad(input=x, pad=(padding, padding, padding, padding), mode='constant',
                                        value=padding_value)
    h_out = int((height + 2 * padding - kernel_size) / stride + 1)
    w_out = int((width + 2 * padding - kernel_size) / stride + 1)
    out = torch.zeros((int(batch_size), int(channel_out), int(h_out), int(w_out)))
    for b in range(batch_size):
        for ch in range(channel_out):
            for i in range(h_out):
                for j in range(w_out):
                    out[b,ch,i,j]=torch.sum(torch.eq(x_pad[b,:,i*stride:i*stride+kernel_size,
                                                     j*stride:j*stride+kernel_size],w[ch,:,:,:]))
    return out

def param_gen(gamma,beta,mean,var,bias,channel,kernel_size):
    gamma_1=2*gamma/torch.sqrt(var)
    beta_1=(bias-mean-channel*kernel_size*kernel_size)/torch.sqrt(var)*gamma+beta
    return gamma_1,beta_1

if __name__ == '__main__':
    setup_seed(1)  # 随机种子——训练结果可复现

    train_dataset = torchvision.datasets.MNIST(root='../../data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='../../data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=128,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=128,
                                              shuffle=False)

    print('******Initializing model******')
    # ******************** 在model的量化卷积中同时量化A(特征)和W(模型参数) ************************
    model = Net(A=2, W=2)
    best_acc = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    # cpu、gpu
    model.to(device)
    print(model)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

    # 训练模型
    for epoch in range(1, 10):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()

    param = model.state_dict()

    WeightBin = weight_tnn_bin(2)

    # 浮点卷积层
    Wc1 = param['tnn_bin.0.weight'].cpu()
    bc1 = param['tnn_bin.0.bias'].cpu()
    # BN层
    bn1_mean = param['tnn_bin.1.running_mean'].cpu()
    bn1_var = param['tnn_bin.1.running_var'].cpu()
    bn1_gamma = param['tnn_bin.1.weight'].cpu()
    bn1_beta = param['tnn_bin.1.bias'].cpu()
    # 二值卷积层1，2
    BWc1 = WeightBin.forward(param['tnn_bin.3.tnn_bin_conv.weight']).cpu()
    BWc1 = (BWc1+1)/2
    Bbc1 = param['tnn_bin.3.tnn_bin_conv.bias'].cpu()
    bn2_mean = param['tnn_bin.3.bn.running_mean'].cpu()
    bn2_var = param['tnn_bin.3.bn.running_var'].cpu()
    bn2_gamma = param['tnn_bin.3.bn.weight'].cpu()
    bn2_beta = param['tnn_bin.3.bn.bias'].cpu()

    gamma2,beta2=param_gen(gamma=bn2_gamma,beta=bn2_beta,mean=bn2_mean,var=bn2_var,bias=Bbc1,channel=16,kernel_size=5)

    BWc2 = WeightBin.forward(param['tnn_bin.4.tnn_bin_conv.weight']).cpu()
    BWc2 = (BWc2+1)/2
    Bbc2 = param['tnn_bin.4.tnn_bin_conv.bias'].cpu()
    bn3_mean = param['tnn_bin.4.bn.running_mean'].cpu()
    bn3_var = param['tnn_bin.4.bn.running_var'].cpu()
    bn3_gamma = param['tnn_bin.4.bn.weight'].cpu()
    bn3_beta = param['tnn_bin.4.bn.bias'].cpu()

    gamma3, beta3 = param_gen(gamma=bn3_gamma, beta=bn3_beta, mean=bn3_mean, var=bn3_var, bias=Bbc2, channel=32,
                              kernel_size=5)
    # 二值卷积层3，4
    BWc3 = WeightBin.forward(param['tnn_bin.6.tnn_bin_conv.weight']).cpu()
    BWc3 = (BWc3+1)/2
    Bbc3 = param['tnn_bin.6.tnn_bin_conv.bias'].cpu()
    bn4_mean = param['tnn_bin.6.bn.running_mean'].cpu()
    bn4_var = param['tnn_bin.6.bn.running_var'].cpu()
    bn4_gamma = param['tnn_bin.6.bn.weight'].cpu()
    bn4_beta = param['tnn_bin.6.bn.bias'].cpu()

    gamma4, beta4 = param_gen(gamma=bn4_gamma, beta=bn4_beta, mean=bn4_mean, var=bn4_var, bias=Bbc3, channel=32,
                              kernel_size=5)

    BWc4 = WeightBin.forward(param['tnn_bin.7.tnn_bin_conv.weight']).cpu()
    BWc4 = (BWc4+1)/2
    Bbc4 = param['tnn_bin.7.tnn_bin_conv.bias'].cpu()
    bn5_mean = param['tnn_bin.7.bn.running_mean'].cpu()
    bn5_var = param['tnn_bin.7.bn.running_var'].cpu()
    bn5_gamma = param['tnn_bin.7.bn.weight'].cpu()
    bn5_beta = param['tnn_bin.7.bn.bias'].cpu()

    gamma5, beta5 = param_gen(gamma=bn5_gamma, beta=bn5_beta, mean=bn5_mean, var=bn5_var, bias=Bbc4, channel=64,
                              kernel_size=5)

    Wc1.numpy().tofile("param\\Wc1.bin")
    bc1.numpy().tofile("param\\bc1.bin")
    bn1_gamma.numpy().tofile("param\\bn1_gamma.bin")
    bn1_beta.numpy().tofile("param\\bn1_beta.bin")
    bn1_mean.numpy().tofile("param\\bn1_mean.bin")
    bn1_var.numpy().tofile("param\\bn1_var.bin")
    BWc1.numpy().tofile("param\\BWc1.bin")
    BWc2.numpy().tofile("param\\BWc2.bin")
    BWc3.numpy().tofile("param\\BWc3.bin")
    BWc4.numpy().tofile("param\\BWc4.bin")
    gamma2.numpy().tofile("param\\gamma2.bin")
    beta2.numpy().tofile("param\\beta2.bin")
    gamma3.numpy().tofile("param\\gamma3.bin")
    beta3.numpy().tofile("param\\beta3.bin")
    gamma4.numpy().tofile("param\\gamma4.bin")
    beta4.numpy().tofile("param\\beta4.bin")
    gamma5.numpy().tofile("param\\gamma5.bin")
    beta5.numpy().tofile("param\\beta5.bin")

    img=np.zeros((10000,1,28,28),dtype=np.float32)
    label=np.zeros((10000,),dtype=np.float32)
    for batch_idx, (data, target) in enumerate(test_loader):
        #print(batch_idx,data.size())
        img[batch_idx*128:batch_idx*128+data.size()[0],:,:,:]=data.cpu().numpy()
        label[batch_idx*128:batch_idx*128+target.size()[0]]=target.cpu().numpy().astype(np.float32)
    img.tofile("param\\img.bin")
    label.tofile("param\\label.bin")

    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):

        x = F.conv2d(data, Wc1, bias=bc1, stride=1, padding=2)
        x = F.batch_norm(x, running_mean=bn1_mean, running_var=bn1_var, weight=bn1_gamma,bias=bn1_beta)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = (torch.sign(x)+1)/2
        x = my_conv(x,BWc1,kernel_size=5,stride=1,padding=2,padding_value=0)
        x=x*gamma2.view(1,-1,1,1)+beta2.view(1,-1,1,1)

        x = (torch.sign(x)+1)/2
        x = my_conv(x, BWc2, kernel_size=5, stride=1, padding=2, padding_value=0)
        x = x * gamma3.view(1, -1, 1, 1) + beta3.view(1, -1, 1, 1)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = (torch.sign(x)+1)/2
        x = my_conv(x, BWc3, kernel_size=5, stride=1, padding=2, padding_value=0)
        x = x * gamma4.view(1, -1, 1, 1) + beta4.view(1, -1, 1, 1)

        x = (torch.sign(x)+1)/2
        x = my_conv(x, BWc4, kernel_size=5, stride=1, padding=2, padding_value=0)
        x = x * gamma5.view(1, -1, 1, 1) + beta5.view(1, -1, 1, 1)
        x=torch.relu(x)

        x = F.avg_pool2d(x, kernel_size=7)

        output = torch.argmax(x, axis=1)
        for i in range(data.size(0)):
            if target[i] == output[i]:
                correct += 1
        print(correct/data.size(0))
        correct=0







