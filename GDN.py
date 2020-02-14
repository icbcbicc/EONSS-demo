import torch
import torch.nn as nn
from torch.autograd import Function


class GdnFunction(Function):

    @staticmethod
    def forward(ctx, x, gamma, beta):
        ctx.save_for_backward(x, gamma, beta)
        n, c, h, w = list(x.size())
        tx = x.permute(0, 2, 3, 1).contiguous()
        tx = tx.view(-1, c)
        tx2 = tx * tx
        denominator = tx2.mm(gamma) + beta
        ty = tx / torch.sqrt(denominator)
        y = ty.view(n, h, w, c)
        y = y.permute(0, 3, 1, 2).contiguous()
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, gamma, beta = ctx.saved_variables
        n, c, h, w = list(grad_output.size())
        tx = x.permute(0, 2, 3, 1).contiguous()
        tx = tx.view(-1, c)
        tx2 = tx * tx
        denominator = tx2.mm(gamma) + beta

        tdzdy = grad_output.permute(0, 2, 3, 1).contiguous()
        tdzdy = tdzdy.view(-1, c)
        gy = (tdzdy * torch.pow(denominator, -0.5) - (tdzdy * tx *
              torch.pow(denominator, -1.5)).mm(gamma.t()) * tx)
        gy = gy.view(n, h, w, c)
        grad_input = gy.permute(0, 3, 1, 2).contiguous()
        tmp = -0.5 * torch.pow(denominator, -1.5) * tx * tdzdy
        grad_beta = torch.sum(tmp, 0)
        grad_gamma = tx2.t().mm(tmp)
        return grad_input, grad_gamma, grad_beta


class Gdn(nn.Module):
    def __init__(self, input_channel):
        super(Gdn, self).__init__()
        self.input_channel = input_channel
        self.gamma = nn.Parameter(torch.Tensor(input_channel, input_channel))
        self.beta = nn.Parameter(torch.Tensor(input_channel))

    def forward(self, input):
        return GdnFunction.apply(input, self.gamma, self.beta)

    def __str__(self):
        return self.__class__.__name__ + '(gamma_size=(%d, %d), beta_size=(%d))' %\
               (self.gamma.size()[0], self.gamma.size()[1], self.beta.size()[0])

    __repr__ = __str__
