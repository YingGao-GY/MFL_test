import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)


class SelfAttention(nn.Module):

    def __init__(self, in_dim,activation=F.relu):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8 , kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        init_conv(self.f)
        init_conv(self.g)

    def forward(self, x, t):
        # m_batchsize, C, width, height = x.size()  # 4*32*64*64
        # context = conv1x1(t.size(1), width).cuda()
        # tt = t.unsqueeze(3)
        # w = context(tt)  # bz x width x t_height
        # w = w.squeeze(3)
        # image = conv1x1(C, 1).cuda()
        # w = torch.bmm(torch.transpose(w, 1, 2).contiguous(), image(x).squeeze(1))  # bz x t_height x height
        # w = self.softmax(w)
        # w, _ = torch.max(w, 2)
        # w = w.unsqueeze(1).repeat(1, t.size(1), 1)
        # out = w.mul(t)
        # out = 0.85 * out + 0.9 * t
        #
        # return out
        m_batchsize, C, width, height = x.size()

        f = self.f(x).view(m_batchsize, -1, width * height)
        g = self.g(x).view(m_batchsize, -1, width * height)

        attention = torch.bmm(f.permute(0, 2, 1), g)
        attention = self.softmax(attention)

        out = torch.bmm(attention, t)
        out = self.gamma * out + t

        return out
