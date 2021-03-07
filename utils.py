import torch

def my_conv(x, w, kernel_size, stride, padding, padding_value):
    batch_size, channel_in, height, width = x.size()
    channel_out, channel_in, kx, ky = w.size()
    x_pad = torch.nn.functional.pad(input=x, pad=(padding, padding, padding, padding), mode='constant',
                                        value=padding_value)
    h_out = int((height + 2 * padding - kernel_size) / stride + 1)
    w_out = int((width + 2 * padding - kernel_size) / stride + 1)
    out = torch.zeros((int(batch_size), int(channel_out), int(h_out), int(w_out))).cuda()
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

def threshold_func(x,T,S):
    b,c,h,w=x.size()
    for n in range(b):
        for m in range(c):
            for i in range(h):
                for j in range(w):
                    if (x[n,m,i,j]>=T[m] and S[m]==1) or (x[n,m,i,j]<T[m] and S[m]==-1):
                        x[n,m,i,j] = 1
                    else:
                        x[n,m,i,j] = 0
    return x