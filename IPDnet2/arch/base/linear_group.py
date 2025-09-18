import torch
from torch import nn, Tensor
from torch.nn import *
import math


class LinearGroup(nn.Module):

    def __init__(self, in_features: int, out_features: int, num_groups: int, bias: bool = True) -> None:
        super(LinearGroup, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight = Parameter(torch.empty((num_groups, out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # same as linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """shape [..., group, feature]"""
        x = torch.einsum("...gh,gkh->...gk", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, num_groups={self.num_groups}, bias={True if self.bias is not None else False}"


import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.init as init
import math

class LinearGroupSharedWeight(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_groups: int, bias: bool = True) -> None:
        super(LinearGroupSharedWeight, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # 权重矩阵形状为 [out_features, in_features]，所有组共享这个权重
        self.weight = Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 使用与标准线性层相同的初始化方法
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """形状 [..., group, feature]"""
        # 扩展 x 的维度，使其与权重矩阵相乘
        x = torch.einsum("...gh,kh->...gk", x, self.weight)
        
        if self.bias is not None:
            # 在每个组的输出中添加偏置
            x = x + self.bias
        
        return x

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, num_groups={self.num_groups}, bias={True if self.bias is not None else False}"



class Conv1dGroup(nn.Module):

    def __init__(self, in_features: int, out_features: int, num_groups: int, kernel_size: int, bias: bool = True) -> None:
        super(Conv1dGroup, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.kernel_size = kernel_size

        self.weight = Parameter(torch.empty((num_groups, out_features, in_features, kernel_size)))
        if bias:
            self.bias = Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # same as linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """shape [batch, time, group, feature]"""
        (B, T, G, F), K = x.shape, self.kernel_size
        x = x.permute(0, 2, 3, 1).reshape(B * G * F, 1, 1, T)  # [B*G*F,1,1,T]
        x = torch.nn.functional.unfold(x, kernel_size=(1, K), padding=(0, K // 2))  # [B*G*F,K,T]
        x = x.reshape(B, G, F, K, T)
        x = torch.einsum("bgfkt,gofk->btgo", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, num_groups={self.num_groups}, kernel_size={self.kernel_size}, bias={True if self.bias is not None else False}"
