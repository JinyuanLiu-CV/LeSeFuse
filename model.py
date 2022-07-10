import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class Cell(nn.Module):

    def __init__(self, genotype, C_prev, C):
        super(Cell, self).__init__()
        self.preprocess = ReLUConvBN(C_prev, C, 1, 1, 0)
        op_names, indices = zip(*genotype.cell)
        concat = genotype.cell_concat
        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](C, C)
            self._ops += [op]
        self._indices = indices

    def forward(self, s):
        s = self.preprocess(s)
        states = [s]
        h = states[0]
        op = self._ops[0]
        s = op(h)
        states += [s]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i + 1]
            op2 = self._ops[2 * i + 2]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class FM(nn.Module):

    def __init__(self, C, genotype1, genotype2, genotype3):
        super(FM, self).__init__()
        self._inC = C  # 4
        self.cell_ir = Cell(genotype1, 1, C)
        self.cell_vis = Cell(genotype2, 1, C)
        self.cell_fu = Cell(genotype3, C*8, C//2)
        self.conv1x1 = nn.Conv2d(C*2, 1, 1, padding=0, bias=True)

    def forward(self, input_ir, input_vis):
        feature_ir = self.cell_ir(input_ir)
        feature_vis = self.cell_vis(input_vis)
        feature_fusion = self.cell_fu(torch.cat([feature_ir, feature_vis], dim=1))
        output = self.conv1x1(feature_fusion)
        return output

