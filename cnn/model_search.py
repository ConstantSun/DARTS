import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
import sys

from operations import FactorizedReduce
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

import numpy as np

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))  # sum up all operations with their corresponding alpha given a Tensor input x.


class Cell(nn.Module):
  """
  Including 6 nodes and 1 output node( only concate the last 4 nodes)
              & the first 2 nodes are outputs from the 2 previous cells
  """

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    #                 4          4
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)    # implementation of this func is pretty weird !
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps       #       4
    self._multiplier = multiplier #   4

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()

    for i in range(self._steps):  # 4
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3): # C = 16 (init channel),
               #    16,    10          8                                                                     # num_classes = 10, 
                                                                                                   # layers (number of cells) = 8.
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C # 3*16 = 48
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),  # C_curr = 48
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    #  48            48    16
    self.cells = nn.ModuleList()

    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      #            4        4          48           48       48      F              F          

      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)

    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      # weights size : (14, 8) 
      s0, s1 = s1, cell(s0, s1, weights)

    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i)) # k = 14
    num_ops = len(PRIMITIVES)                                  # num_ops = 8

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True) # (14, 8)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True) # (14, 8)

    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


# net = Network(16, 10, 8, nn.CrossEntropyLoss())
# old_stdout = sys.stdout

# log_file = open("log/darts_layers.log","w")

# sys.stdout = log_file
# print(net)

# sys.stdout = old_stdout

# log_file.close()

# # pytorch to onnx
# net.eval()
# net.to("cuda")

# model_name = "darts_cifar10"
# model_path = f"./onnx_model/{model_name}.onnx"

# dummy_input = torch.randn(1, 3, 32, 32).to("cuda")
# # dummy_input = torch.randn(1, 3, 480, 640).to("cuda") #if input size is 640*480

# res =  net(dummy_input)
# print(res)
# # with torch.no_grad():
# #   torch.onnx.export(net, dummy_input, model_path, verbose=False, input_names=['input'], output_names=['scores'])


# test_cell = Cell(4, 4, 48, 48, 10, False, False)
# # print(test_cell)

# old_stdout = sys.stdout
# log_file = open("log/cell_arch.log","w")

# sys.stdout = log_file
# print(test_cell)

# sys.stdout = old_stdout

# log_file.close()

# t = MixedOp(64, 1)
# print("._ops : \n-----------------------")
# print(t._ops)


net = Network(16, 10, 8, nn.CrossEntropyLoss())
# print("net parameters: ")
# for i, value in enumerate(net.parameters()):
  
#   if i >= 600 and i <= 700:
#     print(i, " : ", value.shape)

_c = 0
for k, v in net.named_parameters():
  print(k, " : " , v.size())
  v_length = np.prod(v.size())
  print("v_length : ", v_length)
  _c += 1
  if _c > 20:  
    break 
