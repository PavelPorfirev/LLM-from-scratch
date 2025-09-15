import math, torch, torch.nn.functional as F

def linear(x, W, b=None):
    y = x.matmul(W.t())
    if b is not None:
        y = y + b
    return y

def layer_norm(x, weight, bias, eps=1e-5):
    return F.layer_norm(x, x.shape[-1:], weight, bias, eps=eps)

def gelu(x):
    return F.gelu(x)
