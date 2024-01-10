from .baseOptim import BaseOptim
from ..utils import colorText, importModule


OptimRegistry = {}


def registerOptimizer(Name: str):
    def registerOptimizerCls(Cls):
        if Name in OptimRegistry:
            raise ValueError("Cannot register duplicate optimizer ({})".format(Name))

        if not issubclass(Cls, BaseOptim):
            raise ValueError(
                "Optimizer ({}: {}) must extend BaseOptim".format(Name, Cls.__name__)
            )

        OptimRegistry[Name] = Cls
        return Cls

    return registerOptimizerCls


def buildOptimizer(NetParam, opt) -> BaseOptim:
    optimizer = None
    
    if opt.optim in OptimRegistry:
        optimizer = OptimRegistry[opt.optim](opt, NetParam)
    else:
        SupList = list(OptimRegistry.keys())
        SupStr = (
            "Optimizer ({}) not yet supported. \n Supported optimizers are:".format(
                opt.optim
            )
        )
        for i, Name in enumerate(SupList):
            SupStr += "\n\t {}: {}".format(i, colorText(Name))

    return optimizer


# automatically import the optimizers
importModule(__file__, RelativePath="lib.optim.")
