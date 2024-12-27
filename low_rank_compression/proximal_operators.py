import torch
from typing import List


def nuclear_norm_proximal_op(input, lr, multiplier):
    U, S, V = torch.svd(input)
    S_opt = torch.clamp(S - lr * multiplier, min=0)
    desired_param = U @ S_opt.diag() @ V.t()
    return desired_param


class NuclearNormApplier:
    def __init__(self, params: List[torch.Tensor], lr: float, multiplier: float):
        self.params = params
        self.lr = lr
        self.multiplier = multiplier

    def step(self):
        for param in self.params:
            with torch.no_grad():
                param.copy_(nuclear_norm_proximal_op(param, self.lr, self.multiplier))


def singular_valuess_entropy_proximal_op(input, lr):
    _, singular_values, _ = torch.svd(input)
    smx = torch.nn.functional.softmax(singular_values, dim=-1)
    ent = -torch.sum(smx * torch.log(smx.clamp_min(1e-12)))
    input_grad_wrt_entropy = torch.autograd.grad(ent, input, retain_graph=True)[0]
    desired_param = input - input_grad_wrt_entropy * lr
    return desired_param


class SingularValuesEntropyApplier:
    def __init__(self, params: List[torch.Tensor], lr: float):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            ret = singular_valuess_entropy_proximal_op(param, self.lr)
            with torch.no_grad():
                param.copy_(ret)
