import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ReduceMetric(nn.Module):
  def __init__(self, mode):
    super().__init__()
    if mode == 'avg':
      self.forward = self.avg_forward
    elif mode == 'max':
      self.forward = self.max_forward
    elif mode == 'maxavg':
      self.maxavg_activation = MaxAvgGlobalActivation()
      self.forward = self.maxavg_forward
    else:
      raise NotImplementedError

  def maxavg_forward(self, x):
    return self.maxavg_activation(x)

  def max_forward(self, x):
    return x.max(-1)[0]

  def avg_forward(self, x):
    return x.mean(-1)

class ConstrainedLinear(nn.Linear):
  def forward(self, x):
    return F.linear(x, torch.min(self.weight ** 2, torch.abs(self.weight)))

class MaxReLUPairwiseActivation(nn.Module):
  def __init__(self, num_features):
    super().__init__()
    self.weights = nn.Parameter(torch.zeros(1, num_features))
    self.avg_pool = nn.AvgPool1d(2, 2)

  def forward(self, x):
    x = x.unsqueeze(1)
    max_component = F.max_pool1d(x, 2)
    relu_component = F.avg_pool1d(F.relu(x * F.softplus(self.weights)), 2)
    output = torch.cat((max_component, relu_component), dim=-1).squeeze(1)
    return output

class ConcaveActivation(nn.Module):
  def __init__(self, num_features, concave_activation_size):
    super().__init__()
    assert concave_activation_size > 1

    self.bs_nonzero = nn.Parameter(1e-3 * torch.randn((1, num_features, concave_activation_size - 1)) - 1)
    self.bs_zero    = torch.zeros((1, num_features, 1))
    self.ms = nn.Parameter(1e-3 * torch.randn((1, num_features, concave_activation_size)))

  def forward(self, x):
    bs = torch.cat((F.softplus(self.bs_nonzero), self.bs_zero), -1)
    ms = 2 * torch.sigmoid(self.ms)
    x = x.unsqueeze(-1)

    x = x * ms + bs
    return x.min(-1)[0]

class DeepNormMetric(nn.Module):
  def __init__(self, num_features, layers, activation=nn.ReLU, concave_activation_size=None, mode='avg', symmetric=False):
    super().__init__()
    self.num_features = num_features

    assert len(layers) >= 2

    self.Us = nn.ModuleList([nn.Linear(num_features, layers[0], bias=False)])
    self.Ws = nn.ModuleList([])

    for in_features, out_features in zip(layers[:-1], layers[1:]):
      self.Us.append(nn.Linear(num_features, out_features, bias=False))
      self.Ws.append(ConstrainedLinear(in_features, out_features, bias=False))

    self.activation = activation()
    self.output_activation = ConcaveActivation(layers[-1], concave_activation_size) if concave_activation_size else nn.Identity()
    self.reduce_metric = ReduceMetric(mode)

    self.symmetric = symmetric

  def _asym_fwd(self, h):
    h1 = self.Us[0](h)
    for U, W in zip(self.Us[1:], self.Ws):
      h1 = self.activation(W(h1) + U(h))
    return h1

  def forward(self, x, y):
    h = x - y
    if self.symmetric:
      h = self._asym_fwd(h) + self._asym_fwd(-h)
    else:
      h = self._asym_fwd(-h)
    h = self.activation(h)
    return self.reduce_metric(h)
