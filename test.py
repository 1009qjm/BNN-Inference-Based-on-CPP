import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

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

Wc1=torch.from_numpy(np.fromfile("param\\Wc1.bin",dtype=np.float32)).view(16,1,5,5)
bc1=torch.from_numpy(np.fromfile("param\\bc1.bin",dtype=np.float32))
bn1_gamma=torch.from_numpy(np.fromfile("param\\bn1_gamma.bin",dtype=np.float32))
bn1_beta=torch.from_numpy(np.fromfile("param\\bn1_beta.bin",dtype=np.float32))
bn1_mean=torch.from_numpy(np.fromfile("param\\bn1_mean.bin",dtype=np.float32))
bn1_var=torch.from_numpy(np.fromfile("param\\bn1_var.bin",dtype=np.float32))
BWc1=torch.from_numpy(np.fromfile("param\\BWc1.bin",dtype=np.float32)).view(32,16,5,5)
BWc2=torch.from_numpy(np.fromfile("param\\BWc2.bin",dtype=np.float32)).view(32,32,5,5)
BWc3=torch.from_numpy(np.fromfile("param\\BWc3.bin",dtype=np.float32)).view(64,32,5,5)
BWc4=torch.from_numpy(np.fromfile("param\\BWc4.bin",dtype=np.float32)).view(10,64,5,5)
gamma2=torch.from_numpy(np.fromfile("param\\gamma2.bin",dtype=np.float32))
beta2=torch.from_numpy(np.fromfile("param\\beta2.bin",dtype=np.float32))
gamma3=torch.from_numpy(np.fromfile("param\\gamma3.bin",dtype=np.float32))
beta3=torch.from_numpy(np.fromfile("param\\beta3.bin",dtype=np.float32))
gamma4=torch.from_numpy(np.fromfile("param\\gamma4.bin",dtype=np.float32))
beta4=torch.from_numpy(np.fromfile("param\\beta4.bin",dtype=np.float32))
gamma5=torch.from_numpy(np.fromfile("param\\gamma5.bin",dtype=np.float32))
beta5=torch.from_numpy(np.fromfile("param\\beta5.bin",dtype=np.float32))




test_dataset = torchvision.datasets.MNIST(root='../../data',
                                              train=False,
                                              transform=transforms.ToTensor())
# Data loader
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False)

correct = 0
for batch_idx, (data, target) in enumerate(test_loader):

    x = F.conv2d(data, Wc1, bias=bc1, stride=1, padding=2)
    x = F.batch_norm(x, running_mean=bn1_mean, running_var=bn1_var, weight=bn1_gamma,bias=bn1_beta)
    x = F.max_pool2d(x, kernel_size=2, stride=2)

    x = (torch.sign(x)+1)/2
    x = my_conv(x,BWc1,kernel_size=5,stride=1,padding=2,padding_value=0)
    x = x * gamma2.view(1,-1,1,1)+beta2.view(1,-1,1,1)

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

