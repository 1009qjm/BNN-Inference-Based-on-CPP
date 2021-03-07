from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from model import *
from utils import *
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

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item(),
                optimizer.param_groups[0]['lr']))
    return


# 模型测试
def test():
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

if __name__ == '__main__':
    setup_seed(1)
    is_train=False
    train_dataset = torchvision.datasets.MNIST(root='../../data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='../../data',
                                              train=False,
                                              transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=128,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=128,
                                              shuffle=False)

    model = Net(A=2, W=2)
    if is_train:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

    if is_train:
        for epoch in range(1, 20):
            adjust_learning_rate(optimizer, epoch)
            train(epoch)
            test()
        param = model.state_dict()
        torch.save(param, "param.pth")
        for k, v in param.items():
            print(k, v.size())
    else:
        model.load_state_dict(torch.load('param.pth'))
        test()

        param = model.state_dict()
        WeightBin = weight_tnn_bin(2)

        img=np.zeros((10000,1,28,28),dtype=np.float32)
        label=np.zeros((10000,),np.float32)
        for batch_idx, (data, target) in enumerate(test_loader):
            img[128*batch_idx:128*batch_idx+128,:,:,:]=data.cpu().numpy()
            label[128*batch_idx:128*batch_idx+128]=target.cpu().numpy().astype(np.float32)
        img.tofile("image.bin")
        label.tofile("label.bin")
        #
        Wf1=param['tnn_bin.0.weight']
        bf1=param['tnn_bin.0.bias']
        gamma1=param['tnn_bin.1.weight']
        beta1=param['tnn_bin.1.bias']
        mean1=param['tnn_bin.1.running_mean']
        var1=param['tnn_bin.1.running_var']
        #
        Wf2 =param['tnn_bin.3.tnn_bin_conv.weight']
        bf2 =param['tnn_bin.3.tnn_bin_conv.bias']
        gamma2 =param['tnn_bin.3.bn.weight']
        beta2 =param['tnn_bin.3.bn.bias']
        mean2 =param['tnn_bin.3.bn.running_mean']
        var2 =param['tnn_bin.3.bn.running_var']
        #
        Wf3 = param['tnn_bin.4.tnn_bin_conv.weight']
        bf3 = param['tnn_bin.4.tnn_bin_conv.bias']
        gamma3 = param['tnn_bin.4.bn.weight']
        beta3 = param['tnn_bin.4.bn.bias']
        mean3 = param['tnn_bin.4.bn.running_mean']
        var3 = param['tnn_bin.4.bn.running_var']
        #
        Wf4 = param['tnn_bin.6.tnn_bin_conv.weight']
        bf4 = param['tnn_bin.6.tnn_bin_conv.bias']
        gamma4 = param['tnn_bin.6.bn.weight']
        beta4 = param['tnn_bin.6.bn.bias']
        mean4 = param['tnn_bin.6.bn.running_mean']
        var4 = param['tnn_bin.6.bn.running_var']
        #
        Wf5 = param['tnn_bin.7.tnn_bin_conv.weight']
        bf5 = param['tnn_bin.7.tnn_bin_conv.bias']
        gamma5 = param['tnn_bin.7.bn.weight']
        beta5 = param['tnn_bin.7.bn.bias']
        mean5 = param['tnn_bin.7.bn.running_mean']
        var5 = param['tnn_bin.7.bn.running_var']

        Wb2 = (WeightBin.forward(Wf2)+1)/2
        Wb3 = (WeightBin.forward(Wf3)+1)/2
        Wb4 = (WeightBin.forward(Wf4)+1)/2
        Wb5 = (WeightBin.forward(Wf5)+1)/2

        g2 = gamma2/torch.sqrt(var2)
        b2 = beta2-mean2*gamma2/torch.sqrt(var2)
        g3, b3 = param_gen(gamma3, beta3, mean3, var3, bf2, 16, kernel_size=5)
        g4, b4 = param_gen(gamma4, beta4, mean4, var4, bf3, 32, kernel_size=5)
        g5, b5 = param_gen(gamma5, beta5, mean5, var5, bf4, 32, kernel_size=5)

        Wb2 = Wb2.cuda()
        Wb3 = Wb3.cuda()
        Wb4 = Wb4.cuda()
        Wb5 = Wb5.cuda()

        g2, b2 = g2.cuda(), b2.cuda()
        g3, b3 = g3.cuda(), b3.cuda()
        g4, b4 = g4.cuda(), b4.cuda()
        g5, b5 = g5.cuda(), b5.cuda()

        S2 = torch.sign(g2)
        T2 = -b2 / g2
        S3 = torch.sign(g3)
        T3 = -b3 / g3
        S4 = torch.sign(g4)
        T4 = -b4 / g4
        S5 = torch.sign(g5)
        T5 = -b5 / g5
        #存储参数
        Wf1.cpu().numpy().tofile("Wf1.bin")
        bf1.cpu().numpy().tofile("bf1.bin")
        gamma1.cpu().numpy().tofile("gamma1.bin")
        beta1.cpu().numpy().tofile("beta1.bin")
        mean1.cpu().numpy().tofile("mean1.bin")
        var1.cpu().numpy().tofile("var1.bin")
        Wb2.cpu().numpy().tofile("Wb2.bin")
        S2.cpu().numpy().tofile("S2.bin")
        T2.cpu().numpy().tofile("T2.bin")
        Wb3.cpu().numpy().tofile("Wb3.bin")
        S3.cpu().numpy().tofile("S3.bin")
        T3.cpu().numpy().tofile("T3.bin")
        Wb4.cpu().numpy().tofile("Wb4.bin")
        S4.cpu().numpy().tofile("S4.bin")
        T4.cpu().numpy().tofile("T4.bin")
        Wb5.cpu().numpy().tofile("Wb5.bin")
        S5.cpu().numpy().tofile("S5.bin")
        T5.cpu().numpy().tofile("T5.bin")

        #推理，B-A层融合为阈值函数
        correct=0
        for batch_idx, (data, target) in enumerate(test_loader):
            data,target = data.cuda(),target.cuda()
            x = F.conv2d(data,Wf1,bf1,stride=1,padding=2)                                   #16x28x28
            x = F.batch_norm(x,running_mean=mean1,running_var=var1,weight=gamma1,bias=beta1)
            x = F.max_pool2d(x,kernel_size=2,stride=2)                                      #16x14x14
            #B-A-C
            x = threshold_func(x, T2, S2)
            x = my_conv(x, Wb2, kernel_size=5, stride=1, padding=2, padding_value=0)            #32x14x14
            #B-A-C-P
            x = threshold_func(x,T3,S3)
            x = my_conv(x, Wb3, kernel_size=5, stride=1, padding=2, padding_value=0)       #32x14x14
            x = F.max_pool2d(x,kernel_size=2,stride=2)                                      #32x7x7
            #B-A-C
            x = threshold_func(x,T4,S4)
            x = my_conv(x, Wb4, kernel_size=5, stride=1, padding=2, padding_value=0)       #64x7x7
            #B-A-C
            x = threshold_func(x,T5,S5)
            x = my_conv(x, Wb5, kernel_size=5, stride=1, padding=2, padding_value=0)       #10x7x7

            x = 2*x -64*5*5
            x = torch.relu(x)
            x = F.avg_pool2d(x,kernel_size=7)                                               #10

            output=torch.argmax(x,dim=1)
            for i in range(data.size(0)):
                if target[i]==output[i]:
                    correct+=1
            print(correct/data.size(0))
            correct=0








