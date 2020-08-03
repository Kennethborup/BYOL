import torch
import torch.nn as nn

class EMA(object):
    """
    Exponential Moving Average class.
    
    :param beta: float; decay parameter
    """
    def __init__(self, beta):
        self.beta = beta
        
    def __call__(self, MA, value):
        return MA * self.beta + (1 - self.beta) * value
    

class RandomApply(nn.Module):
    """
    Randomly apply function with probability p.
    
    :param func: function; takes input x (likely augmentation)
    :param p: float; probability of applying func
    """
    def __init__(self, func, p):
        super(RandomApply, self).__init__()
        self.func = func
        self.p = p
        
    def forward(self, x):
        if torch.rand(1)  > self.p:
            return x
        else:
            return self.func(x)
        

class Hook():
    """
    A simple hook class that returns the output of a layer of a model during forward pass.
    """
    def __init__(self):
        self.output = None
        
    def setHook(self, module):
        """
        Attaches hook to model.
        """
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        """
        Saves the wanted information.
        """
        self.output = output
        
    def val(self):
        """
        Return the saved value.
        """
        return self.output