#from torch.nn.modules.container import T
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import models.archs.ops as ops

# Gate mechanism
class Attention_SEblock(nn.Layer):
    def __init__(self, channels, reduction, temperature):  # 64, 4, 1
        super(Attention_SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, 2)
        #self.fc2.bias[0] = 0.1
        #self.fc2.bias[1] = 2
        self.temperature = temperature
        self.channels = channels

    def forward(self, x):
        x = paddle.reshape(self.avg_pool(x),[-1, self.channels])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.gumbel_softmax(x, tau=1, hard=True)
        return x


class Net(nn.Layer):
    def __init__(self, scale, **kwargs):
        super(Net, self).__init__()

        self.scale = scale  # value of scale is scale.
        multi_scale = 0  # value of multi_scale is multi_scale in args.
        kernel_size = 3
        kernel_size1 = 1
        # padding1 = 0
        # padding = 1
        features = 64
        # groups = 1
        channels = 3
        # features1 = 64
        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.conv1 = nn.Sequential(
            nn.Conv2D(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv6 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv7 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv8 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv9 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv10 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())

        # Dynamic Gate
        self.gate = Attention_SEblock(channels=features, reduction=4, temperature=1)
        self.conv7_1 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv8_1 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv9_1 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv10_1 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv7_2 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv8_2 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size1, padding=0, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv9_2 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv10_2 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size1, padding=0, groups=1,
                      bias_attr=False),
            nn.ReLU())

        self.conv11 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv12 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv13 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv14 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())
        self.conv15 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())

        self.conv16 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=1,
                      bias_attr=False),
            nn.ReLU())

        # Upsampling
        self.upsample = ops.UpsampleBlock(64, scale=scale, multi_scale=multi_scale, group=1)
        self.conv17 = nn.Sequential(
            nn.Conv2D(in_channels=features, out_channels=3, kernel_size=kernel_size, padding=1, groups=1, bias_attr=False))

    def forward(self, x, scale):
        x0 = self.sub_mean(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x4_1 = x2 + x4
        x5 = self.conv5(x4_1)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv8(x8)
        x10 = self.conv8(x9)

        decision = self.gate(x6)
        if decision[:, 1].cpu().detach().numpy().all() == 1:
            x7_1 = self.conv7_1(x6)
            x8_1 = self.conv8_1(x7_1)
            x9_1 = self.conv9_1(x8_1)
            x10_1 = self.conv10_1(x9_1)
            x10_3 = x10 + x10_1
        else:
            x7_2 = self.conv7_2(x6)
            x8_2 = self.conv8_2(x7_2)
            x9_2 = self.conv9_2(x8_2)
            x10_2 = self.conv10_2(x9_2)
            x10_3 = x10 + x10_2

        x11 = self.conv11(x10_3)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)

        x15_1 = x15 + x4 + x1
        x16 = self.conv16(x15_1)
        temp = self.upsample(x16, scale=scale)
        x17 = self.conv17(temp)
        out = self.add_mean(x17)

        return out
