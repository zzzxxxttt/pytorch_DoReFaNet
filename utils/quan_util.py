import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 1:
        out = torch.sign(input)
      else:
        n = float(2 ** k - 1)
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply


class quantize_fn(nn.Module):
  def __init__(self, w_bit):
    super(quantize_fn, self).__init__()
    assert w_bit <= 8
    self.w_bit = w_bit
    self.uniform_q = uniform_quantize(k=w_bit)

  def forward(self, x):
    if self.w_bit == 1:
      E = torch.mean(torch.abs(x)).detach()
      weight_q = self.uniform_q(x / E) * E
    else:
      weight = F.tanh(x)
      # weight = weight / torch.mean(torch.abs(weight))
      weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
      weight_q = 2 * self.uniform_q(weight) - 1
    return weight_q


def conv2d_Q_fn(w_bit, order=1):
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
      assert 0 < w_bit <= 8
      assert 0 < order <= 8
      self.w_bit = w_bit
      self.order = order
      self.quantize_fn = quantize_fn(w_bit=w_bit)

    def forward(self, input, order=None):
      weight_q = self.quantize_fn(self.weight)
      for _ in range(self.order if order is None else order - 1):
        weight_q += self.quantize_fn(self.weight - weight_q)
      # print(np.unique(weight_q.detach().numpy()).shape[0])
      return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv2d_Q


def linear_Q_fn(w_bit, order=1):
  class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
      super(Linear_Q, self).__init__(in_features, out_features, bias)
      assert 0 < w_bit <= 8
      assert 0 < order <= 8
      self.w_bit = w_bit
      self.order = order
      self.quantize_fn = quantize_fn(w_bit=w_bit)

    def forward(self, input, order=None):
      weight_q = self.quantize_fn(self.weight)
      for _ in range(self.order - 1 if order is None else order - 1):
        weight_q += self.quantize_fn(self.weight - weight_q)
      return F.linear(input, weight_q, self.bias)

  return Linear_Q


if __name__ == '__main__':
  import numpy as np
  import matplotlib.pyplot as plt

  a = torch.rand(100)
  b = uniform_quantize(4)(a).numpy()

  a = torch.rand(1, 3, 32, 32, requires_grad=True)

  Conv2d = conv2d_Q_fn(w_bit=1, order=4)
  conv = Conv2d(in_channels=3, out_channels=16, kernel_size=1)
  weight_q1 = conv.quantize_fn(conv.weight)
  weight_q2 = weight_q1 + conv.quantize_fn(conv.weight - weight_q1)
  weight_q3 = weight_q2 + conv.quantize_fn(conv.weight - weight_q2)
  weight_q4 = weight_q3 + conv.quantize_fn(conv.weight - weight_q3)
  weight_q5 = weight_q4 + conv.quantize_fn(conv.weight - weight_q4)
  weight_q6 = weight_q5 + conv.quantize_fn(conv.weight - weight_q5)

  # weight_q1 = quantize_fn(w_bit=1)(conv.weight)
  # weight_q2 = quantize_fn(w_bit=2)(conv.weight)
  # weight_q3 = quantize_fn(w_bit=3)(conv.weight)
  # weight_q4 = quantize_fn(w_bit=4)(conv.weight)
  # weight_q5 = quantize_fn(w_bit=5)(conv.weight)

  # weight_q2_ = quantize_fn(w_bit=4)(conv.weight)

  weight_q1 = np.unique(weight_q1.detach().numpy())
  weight_q2 = np.unique(weight_q2.detach().numpy())
  weight_q3 = np.unique(weight_q3.detach().numpy())
  weight_q4 = np.unique(weight_q4.detach().numpy())
  weight_q5 = np.unique(weight_q5.detach().numpy())
  # weight_q2_ = np.unique(weight_q2_.detach().numpy())
  plt.plot(weight_q1, [1] * len(weight_q1), 'C0o')
  plt.plot(weight_q2, [2] * len(weight_q2), 'C1o')
  plt.plot(weight_q3, [3] * len(weight_q3), 'C2o')
  plt.plot(weight_q4, [4] * len(weight_q4), 'C3o')
  plt.plot(weight_q5, [5] * len(weight_q5), 'C4o')
  plt.yticks([1, 2, 3, 4, 5])
  plt.xlabel('value')
  plt.ylabel('bits')
  # plt.plot(weight_q2_, [0] * len(weight_q2_), 'mx')
  plt.show()

  b = conv(a)
  b.retain_grad()
  c = torch.mean(b)
  c.retain_grad()

  c.backward()
  pass
