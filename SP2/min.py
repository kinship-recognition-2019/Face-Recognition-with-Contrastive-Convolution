#coding:utf-8
import torch
from torch.autograd import Variable

class Min(torch.autograd.Function):

    def forward(self, input_):
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        self.save_for_backward(input_)         # 将输入保存起来，在backward时使用
        output = input_.clamp(min=0)           # relu就是截断负数，让所有负数等于0
        return output

    def backward(self, grad_output):
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput就是输入的参数grad_output、
        # 因此只需求relu的导数，在乘以grad_outpu
        input_, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_ < 0] = 0               # 上诉计算的结果就是左式。即ReLU在反向传播中可以看做一个通道选择函数，所有未达到阈值（激活值<0）的单元的梯度都为0
        return grad_input

