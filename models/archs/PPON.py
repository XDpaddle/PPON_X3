import math
import paddle.nn as nn
import paddle

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, groups=groups)

def activation(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

class _ResBlock_32(nn.Layer):
    def __init__(self, nc=64):
        super(_ResBlock_32, self).__init__()
        self.c1 = conv_layer(nc, nc, 3, 1, 1)
        self.d1 = conv_layer(nc, nc//2, 3, 1, 1)  # rate=1
        self.d2 = conv_layer(nc, nc//2, 3, 1, 2)  # rate=2
        self.d3 = conv_layer(nc, nc//2, 3, 1, 3)  # rate=3
        self.d4 = conv_layer(nc, nc//2, 3, 1, 4)  # rate=4
        self.d5 = conv_layer(nc, nc//2, 3, 1, 5)  # rate=5
        self.d6 = conv_layer(nc, nc//2, 3, 1, 6)  # rate=6
        self.d7 = conv_layer(nc, nc//2, 3, 1, 7)  # rate=7
        self.d8 = conv_layer(nc, nc//2, 3, 1, 8)  # rate=8
        self.act = activation('lrelu')
        self.c2 = conv_layer(nc * 4, nc, 1, 1, 1)  # 256-->64

    def forward(self, input):
        output1 = self.act(self.c1(input))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)
        d5 = self.d5(output1)
        d6 = self.d6(output1)
        d7 = self.d7(output1)
        d8 = self.d8(output1)

        add1 = d1 + d2
        add2 = add1 + d3
        add3 = add2 + d4
        add4 = add3 + d5
        add5 = add4 + d6
        add6 = add5 + d7
        add7 = add6 + d8

        combine = paddle.concat([d1, add1, add2, add3, add4, add5, add6, add7], 1)
        output2 = self.c2(self.act(combine))
        output = input + output2*0.2

        return output

class RRBlock_32(nn.Layer):
    def __init__(self):
        super(RRBlock_32, self).__init__()
        self.RB1 = _ResBlock_32()
        self.RB2 = _ResBlock_32()
        self.RB3 = _ResBlock_32()

    def forward(self, input):
        out = self.RB1(input)
        out = self.RB2(out)
        out = self.RB3(out)
        return out*0.2 + input

def upconv_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1, act_type='relu'):
    upsample = nn.Upsample(scale_factor=upscale_factor, mode='nearest')
    conv = conv_layer(in_channels, out_channels, kernel_size, stride)
    act = activation(act_type)
    return sequential(upsample, conv, act)
from collections import OrderedDict
def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Layer):
            modules.append(module)
    return nn.Sequential(*modules)


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding
def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2D(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer
def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2D(nc)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2D(nc)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer
def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):

    padding = get_valid_padding(kernel_size, dilation)  #1
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None #none
    padding = padding if pad_type == 'zero' else 0  #1

    c = nn.Conv2D(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)

class ShortcutBlock(nn.Layer):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr

class PPON_content(nn.Layer):
    def __init__(self, in_nc=3, nf=64, nb=24, out_nc=3, upscale=4, act_type='lrelu'):
        super(PPON_content, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = conv_layer(in_nc, nf, kernel_size=3)  # common
        rb_blocks = [RRBlock_32() for _ in range(nb)]  # L1
        LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = upconv_block

        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]

        HR_conv0 = conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)


        self.CFEM = sequential(fea_conv, ShortcutBlock(sequential(*rb_blocks, LR_conv)))

        self.CRM = sequential(*upsampler, HR_conv0, HR_conv1)  # recon content

    def forward(self, x):
        out_CFEM = self.CFEM(x)
        out_c = self.CRM(out_CFEM)

        return out_c