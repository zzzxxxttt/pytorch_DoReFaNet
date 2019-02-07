import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.quan_util import *


class PreActBlock_conv(nn.Module):
  def __init__(self, in_planes, out_planes, stride=1):
    super(PreActBlock_conv, self).__init__()
    self.bn0 = nn.BatchNorm2d(in_planes)
    self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_planes)
    self.conv1 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

    self.skip_conv = None
    if stride != 1:
      self.skip_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
      self.skip_bn = nn.BatchNorm2d(out_planes)

  def forward(self, x):
    out = F.relu(self.bn0(x))

    if self.skip_conv is not None:
      shortcut = self.skip_conv(out)
      shortcut = self.skip_bn(shortcut)
    else:
      shortcut = x

    out = self.conv0(out)
    out = F.relu(self.bn1(out))
    out = self.conv1(out)
    out += shortcut
    return out


class PreActBlock_conv_Q(nn.Module):
  '''Pre-activation version of the BasicBlock.'''

  def __init__(self, in_planes, out_planes, stride=1):
    super(PreActBlock_conv_Q, self).__init__()
    Conv2d = conv2d_Q_fn(w_bit=1, order=2)

    self.bn0 = nn.BatchNorm2d(in_planes)
    self.conv0 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_planes)
    self.conv1 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

    self.skip_conv = None
    if stride != 1:
      self.skip_conv = self.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
      self.skip_bn = nn.BatchNorm2d(out_planes)

  def forward(self, x):
    out = F.relu(self.bn0(x))

    if self.skip_conv is not None:
      shortcut = self.skip_conv(out)
      shortcut = self.skip_bn(shortcut)
    else:
      shortcut = x

    out = self.conv0(out)
    out = F.relu(self.bn1(out))
    out = self.conv1(out)
    out += shortcut
    return out


class PreActResNet(nn.Module):
  def __init__(self, block, num_units, num_classes):
    super(PreActResNet, self).__init__()
    self.Conv2d = conv2d_Q_fn(w_bit=1, order=2)

    self.in_planes = 16

    self.conv0 = self.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.block1 = self._make_layer(block, 16, num_units[0], stride=1)
    self.block2 = self._make_layer(block, 32, num_units[1], stride=2)
    self.block3 = self._make_layer(block, 64, num_units[2], stride=2)
    self.bn = nn.BatchNorm2d(64)
    self.logit = nn.Linear(64, num_classes)

  def _make_layer(self, block, out_planes, num_units, stride):
    strides = [stride] + [1] * (num_units - 1)
    units = []
    for i, stride in enumerate(strides):
      units.append(block(self.in_planes, out_planes, stride))
      self.in_planes = out_planes
    return nn.Sequential(*units)

  def forward(self, x):
    out = self.conv0(x)
    out = self.block1(out)
    out = self.block2(out)
    out = self.block3(out)
    out = self.bn(out)
    out = out.mean(dim=2).mean(dim=2)
    out = self.logit(out)
    return out


def PreActResNet20_conv(num_classes=10):
  return PreActResNet(PreActBlock_conv, [3, 3, 3], num_classes=num_classes)


def PreActResNet20_conv_Q(num_classes=10):
  return PreActResNet(PreActBlock_conv_Q, [3, 3, 3], num_classes=num_classes)


if __name__ == '__main__':
  features = []


  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)
    features.append(output.data.cpu().numpy())


  net = PreActResNet20_conv()
  for m in net.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      m.register_forward_hook(hook)

  y = net(torch.randn(1, 3, 32, 32))
  print(y.size())
