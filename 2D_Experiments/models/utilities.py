import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn

import operator
from functools import reduce
from functools import partial
import matplotlib.pyplot as plt  

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class UnitGaussianNormalizer:
    def __init__(self, x, eps=1e-5):

        self.mean = torch.mean(x, 0).view(-1)
        self.std = torch.std(x, 0).view(-1)

        self.eps = eps

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.mean) / (self.std + self.eps)
        x = x.view(s)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            std = self.std[sample_idx] + self.eps  # batch * n
            mean = self.mean[sample_idx]

        s = x.size()
        x = x.view(s[0], -1)
        x = (x * std) + mean
        x = x.view(s)
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, pointwise gaussian
class UnitGaussianNormalizer_Siavash(object):
    def __init__(self, x):
        super(UnitGaussianNormalizer_Siavash, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)

    def encode(self, x):
        x = (x - self.mean) / (self.std)
        x = torch.nan_to_num(x, nan=0.0)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx]  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx]  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x


# loss function with rel/abs Lp loss
class LpLoss_Siavash(object):
    def __init__(self, d=2, p=2, damage=False, size_average=True, reduction=True):
        super(LpLoss_Siavash, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.damage = damage
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        if self.damage:
            y_norms = 1.0
        else:
            y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms ** 2.0 / y_norms ** 2.0)
            else:
                return torch.sum(diff_norms ** 2.0 / y_norms ** 2.0)

        return diff_norms ** 2.0 / y_norms ** 2.0

    def __call__(self, x, y):
        return self.rel(x, y)


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1, ] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx // 2, step=1), torch.arange(start=-nx // 2, end=0, step=1)),
                        0).reshape(nx, 1).repeat(1, ny)
        k_y = torch.cat((torch.arange(start=0, end=ny // 2, step=1), torch.arange(start=-ny // 2, end=0, step=1)),
                        0).reshape(1, ny).repeat(nx, 1)
        k_x = torch.abs(k_x).reshape(1, nx, ny, 1).to(x.device)
        k_y = torch.abs(k_y).reshape(1, nx, ny, 1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced == False:
            weight = 1
            if k >= 1:
                weight += a[0] ** 2 * (k_x ** 2 + k_y ** 2)
            if k >= 2:
                weight += a[1] ** 2 * (k_x ** 4 + 2 * k_x ** 2 * k_y ** 2 + k_y ** 4)
            weight = torch.sqrt(weight)
            loss = self.rel(x * weight, y * weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x ** 2 + k_y ** 2)
                loss += self.rel(x * weight, y * weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x ** 4 + 2 * k_x ** 2 * k_y ** 2 + k_y ** 4)
                loss += self.rel(x * weight, y * weight)
            loss = loss / (k + 1)

        return loss


# Error
class Error(object):
    def __init__(self, p=2, field=True, relative=True, mean=False):
        super(Error, self).__init__()

        # Dimension and Lp-norm type are postive
        assert p > 0

        self.p = p
        self.relative = relative
        self.field = field
        self.mean = mean

    def absolute(self, x, y):
        num_examples = x.size()[0]

        if self.field:
            error = torch.abs(x - y)
            if self.mean:
                error = torch.mean(error, 0)
            else:
                error = torch.sum(error, 0)
        else:
            error = torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)
            if self.mean:
                error = torch.mean(error, 0)
            else:
                error = torch.sum(error, 0)
        return error

    def rel(self, x, y):
        num_examples = x.size()[0]

        if self.field:
            error = torch.abs(x - y) / torch.max(torch.abs(y))
            if self.mean:
                error = torch.mean(error, 0)
            else:
                error = torch.sum(error, 0)
        else:
            error = torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1) / torch.norm(
                y.view(num_examples, -1), self.p, 1)
            if self.mean:
                error = torch.mean(error, 0)
            else:
                error = torch.sum(error, 0)
        return error

    def __call__(self, x, y):
        if self.relative:
            return self.rel(x, y)
        else:
            return self.absolute(x, y)


# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul,
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c

def read_map_file(filepath):
    metadata = {}
    map_data = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        metadata['type'] = lines[0].strip().split()[1]
        metadata['height'] = int(lines[1].strip().split()[1])
        metadata['width'] = int(lines[2].strip().split()[1])
        
        # Skip the "map" line
        map_lines = lines[4:]
        for line in map_lines:
            map_data.append(list(line.strip()))
    return metadata, map_data

def display_map(map_data):
    for row in map_data:
        print(' '.join(row))

def visualize_map(map_data):
    height = len(map_data)
    width = len(map_data[0])
    grid = np.zeros((height, width))
    
    for i, row in enumerate(map_data):
        for j, char in enumerate(row):
            if char == '@':
                grid[i, j] = 1

    plt.imshow(grid, cmap='gray')
    plt.show()

def gen_binarymap(map_data):
    height = len(map_data)
    width = len(map_data[0])
    grid = np.zeros((height, width))
    
    for i, row in enumerate(map_data):
        for j, char in enumerate(row):
            if char == '@':
                grid[i, j] = 1

    return  grid            



