class BaseLRScheduler(object):
    def __init__(self, opt, lr_multipliers=None) -> None:
        super().__init__()
        self.round_places = 8
        self.lr_multipliers = lr_multipliers
        self._last_lr = None

    def getLr(self, Epoch: int, CurrIter: int):
        raise NotImplementedError

    def updateLr(self, optimizer, Epoch: int, CurrIter: int):
        lr = self.getLr(Epoch, CurrIter)
        lr = max(0.0, lr)
        self._last_lr = lr
        if self.lr_multipliers is not None:
            assert len(self.lr_multipliers) == len(optimizer.param_groups)
            for g_id, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = round(lr * self.lr_multipliers[g_id], self.round_places)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = round(lr, self.round_places)
        return optimizer

    @staticmethod
    def retrieve_lr(optimizer) -> list:
        lr_list = []
        for param_group in optimizer.param_groups:
            lr_list.append(param_group['lr'])
        return lr_list
    