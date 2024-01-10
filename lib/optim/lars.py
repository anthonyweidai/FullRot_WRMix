from lib2to3.pgen2.token import OP
from typing import Any

import torch
from torch.optim import SGD
from torch.optim.optimizer import Optimizer

from . import registerOptimizer
from .baseOptim import BaseOptim


'''
Lars should use momentum parameters,
not good for simclr for self-supervised leanring, -10%
'''


@registerOptimizer("lars")
class LARS(BaseOptim, Optimizer):
    r"""Implements LARS (Layer-wise Adaptive Rate Scaling).
    Reference: https://github.com/4uiiurz1/pytorch-lars/blob/master/lars.py
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        eta (float, optional): LARS coefficient as used in the paper (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        epsilon (float, optional): epsilon to prevent zero division (default: 0)

    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    
    def __init__(self, opt, NetParam, **kwargs: Any) -> None:
        BaseOptim.__init__(self, opt, **kwargs)
        
        if opt.lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(opt.lr))
        if opt.momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(opt.momentum))
        if opt.weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(opt.weight_decay))

        Defaults = dict(lr=opt.lr, momentum=opt.momentum, eta=opt.eta, dampening=opt.dampening,
                        weight_decay=opt.weight_decay, nesterov=opt.nesterov, epsilon=self.Eps)
        if opt.nesterov and (opt.momentum <= 0 or opt.dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        Optimizer.__init__(self, NetParam, Defaults)
        
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            dampening = group['dampening']
            nesterov = group['nesterov']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                w_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)
                if w_norm * g_norm > 0:
                    local_lr = eta * w_norm / (g_norm +
                        weight_decay * w_norm + epsilon)
                else:
                    local_lr = 1
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-local_lr * group['lr'])

        return loss


@registerOptimizer("lars2")
class LARS(BaseOptim):
    """
    Slight modification of LARC optimizer from https://github.com/NVIDIA/apex/blob/d74fda260c403f775817470d87f810f816f3d615/apex/parallel/LARC.py
    Matches one from SimCLR implementation https://github.com/google-research/simclr/blob/master/lars_optimizer.py

    Args:
        optimizer: Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the adaptive lr. See https://arxiv.org/abs/1708.03888
    """
    def __init__(self, opt, NetParam, **kwargs: Any) -> None:
        BaseOptim.__init__(self, opt, **kwargs)
        self.Sgd = SGD(NetParam, opt.lr, opt.momentum, weight_decay=opt.weight_decay, nesterov=opt.nesterov)
        self.trust_coefficient = 0.001
        self.LayerAdap = True

        self.param_groups = self.Sgd.param_groups
    
    def __getstate__(self):
        return self.Sgd.__getstate__()

    def __setstate__(self, state):
        self.Sgd.__setstate__(state)

    def __repr__(self):
        return self.Sgd.__repr__()

    def state_dict(self):
        return self.Sgd.state_dict()

    def load_state_dict(self, state_dict):
        self.Sgd.load_state_dict(state_dict)

    def zero_grad(self):
        self.Sgd.zero_grad()

    def add_param_group(self, param_group):
        self.Sgd.add_param_group(param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.Sgd.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue

                    if weight_decay != 0:
                        p.grad.data += weight_decay * p.data

                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)
                    adaptive_lr = 1.

                    if param_norm != 0 and grad_norm != 0 and self.LayerAdap:
                        adaptive_lr = self.trust_coefficient * param_norm / grad_norm

                    p.grad.data *= adaptive_lr

        self.Sgd.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.Sgd.param_groups):
            group['weight_decay'] = weight_decays[i]
            