from torch import nn, Tensor
from typing import Any, Tuple

from . import registerSegmentationHead
from .modules import ASPP
from .baseSegHead import BaseSegHead
from ...misc import moduleProfile
from ...layers import BaseConv2d, initWeight


@registerSegmentationHead("deeplabv3")
class DeeplabV3(BaseSegHead):
    """
    This class defines the segmentation head in 
    `DeepLabv3 architecture <https://arxiv.org/abs/1706.05587>`_
    """
    def __init__(self, opt, **kwargs: Any) -> None:
        AtrousRates = (12, 24, 36) # (6, 12, 18)
        OutChannels = 512
        IsSepConv = opt.use_sep_conv
        DropRate = 0.1

        super().__init__(opt, **kwargs)

        self.Aspp = nn.Sequential()
        self.Aspp.add_module(
            name="aspp_layer",
            module=ASPP(
                # self.ModelConfigDict[list(self.ModelConfigDict)[-1]]['out']
                self.callDictbyStage(self.ModelConfigDict, -1),
                OutChannels,
                AtrousRates,
                IsSepConv,
                DropRate,
            ),
        )

        self.Classifier = BaseConv2d(
            OutChannels,
            self.NumClasses,
            kernel_size=1,
            stride=1,
            BNorm=False,
            ActLayer=None,
        )

        if opt.init_weight:
            self.apply(initWeight)

    def forwardSegHead(self, FeaturesTuple: list) -> Tensor:
        # low resolution features
        x = FeaturesTuple[-1]
        # ASPP featues
        x = self.Aspp(x)
        # classify
        return self.Classifier(x)

    def profileModule(self, FeaturesTuple: list) -> Tuple[Tensor, float, float]:
        # Note: Model profiling is for reference only and may contain errors.
        # It relies heavily on the user to implement the underlying functions accurately.

        params, macs = 0.0, 0.0

        x, p, m = moduleProfile(module=self.Aspp, x=FeaturesTuple[-1])
        params += p
        macs += m

        out, p, m = moduleProfile(module=self.Classifier, x=x)
        params += p
        macs += m

        print(
            "{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M".format(
                self.__class__.__name__,
                "Params",
                round(params / 1e6, 3),
                "MACs",
                round(macs / 1e6, 3),
            )
        )
        return out, params, macs
