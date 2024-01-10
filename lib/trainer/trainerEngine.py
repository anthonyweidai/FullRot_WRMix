## Import module
 # path manager
import time
import shutil
from tqdm import tqdm
 # torch module
import torch
import torch.nn.functional as F
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
 # my module
from .baseTrainer import BaseTrainer
from ..model import getModel
from ..metrics import knnPredict
from ..optim import buildOptimizer
from ..optim.scheduler import buildScheduler
from ..loss_fn.utils import mixCriterion
from ..utils import colorText, loadModelWeight, wightFrozen, moveToDevice


class Trainer(BaseTrainer):
    def __init__(self, opt, DataParallel) -> None:
        super().__init__(opt, DataParallel)
        WinCols, _ = shutil.get_terminal_size()
        self.WinCols = 80 if WinCols > 80 else WinCols
        
    def runnerInit(self, Repeat, Split):
        # separate initialisation for inferFeature
        opt = self.opt
        self.splitInit(opt)
        
        self.Repeat = Repeat
        self.Split = Split + opt.num_supplement
            
        Model = getModel(opt=opt)
        Model.to(opt.device)
        
        if self.DataParallel:
            Model = torch.nn.DataParallel(Model)
            # Model = DDP(Model) # self.Optim, grad, and backward should also be changed
            
        if opt.pretrained:
            if opt.freeze_weight == 3:
                Model = loadModelWeight(Model, 2, opt.pretrained_weight, 
                                        opt.pretrained, opt.dist_mode)
            else:
                Model = loadModelWeight(Model, opt.freeze_weight, opt.pretrained_weight, 
                                        opt.pretrained, opt.dist_mode)

        self.Optim = buildOptimizer(Model.parameters(), opt)
        
        self.Model = Model
        
        if 'cosine' in opt.schedular:
            self.Scheduler = buildScheduler(opt, Optimizer=self.Optim)
        else:
            # Scheduler = StepLR(self.Optim, step_size=20, gamma=0.9)
            # Scheduler = ReduceLROnPlateau(self.Optim, mode='min', factor=0.05, patience=5, min_lr=0.0001)
            self.Scheduler = MultiStepLR(self.Optim, milestones=[opt.milestones], gamma=0.1) 
        
        self.dataLoaderSetting(opt)
        
    def run(self, Repeat, Split):
        self.runnerInit(Repeat, Split)
        opt = self.opt # run() is used outside
        self.LrList = []
        
        # Start tranining
        print((' Training%s %s ' % (' and Validating' if self.ValDL else '', 
                                    colorText(str(self.Split + 1)))).center(self.WinCols, '*'))
        for Epoch in range(opt.epochs):
            time.sleep(0.5)  # To prevent possible deadlock during epoch transition
            
            if Epoch == opt.milestones and opt.freeze_weight == 3 and opt.pretrained:
                self.Model = wightFrozen(self.Model, opt.freeze_weight) # From Milestones + 1, stop all layers' weight frizing
            
            self.TrainDL.dataset.callerInit()
            self.training(Epoch)
                
            if opt.lr_decay:
                if "cosine" in opt.schedular:
                    self.Scheduler.step(Epoch)
                else:
                    self.Scheduler.step()
                self.CurrentLrRate = self.Scheduler._last_lr[0] 
            else:
                self.CurrentLrRate = opt.lr
            self.LrList.append(self.CurrentLrRate)
                
            self.bestManager(opt, Epoch, self.Model)
    
    def training(self, Epoch):
        self.Model.train()  # Normalization is different in trainning and evaluation
        self.TrainState.batchInit()
        for Batch in tqdm(self.TrainDL, ncols=self.WinCols - 9, colour='magenta'):
            self.Optim.zero_grad()  # Initialize gradient, preventing accumulation
            
            Batch = moveToDevice(Batch, self.opt.device)
            Img = Batch['image']
            Label = Batch['label']
            
            PredLabel = self.Model(Img)  # prediction
            Loss, FinalPredLabel = mixCriterion(self.opt, self.LossFn, 
                                                Img, PredLabel, Label, Epoch)

            Loss.backward()  # backpropagation
            self.Optim.step()  # optimise model's weight
            
            self.TrainState.batchUpdate(FinalPredLabel, Label, Loss, Epoch)
        
        self.TrainState.update()

        if self.ValDL:
            if '5nn' in self.opt.clsval_mode:
                self.generateFeatureBank()
                
            self.validating(Epoch)

    def validating(self, Epoch):
        self.ValState.batchInit()
        if Epoch >= self.opt.val_start_epoch:
            self.Model.eval()
            with torch.no_grad():
                for Batch in tqdm(self.ValDL, ncols=self.WinCols - 9, colour='magenta'):
                    Batch = moveToDevice(Batch, self.opt.device)
                    Img = Batch['image']
                    Label = Batch['label']
                    # Label = torch.squeeze(Label)
                    
                    if 'linear' in self.opt.clsval_mode:
                        PredLabel = self.Model(Img)  # validation
                        Loss, FinalPredLabel = mixCriterion(self.opt, self.LossFn, 
                                                            Img, PredLabel, Label, Epoch)
                
                        self.ValState.batchUpdate(FinalPredLabel, Label, Loss, Epoch)
                    else:
                        Feature = self.Model.forwardFeatures(Img)
                        Feature = Feature.flatten(1)
                        Feature = F.normalize(Feature, dim=1)
                        
                        FinalPredLabel = knnPredict(self.opt, Feature, 
                                                    self.FeatureBank, self.LabelBank)
                
                        self.ValState.batchUpdate(FinalPredLabel, Label, None, Epoch)
                
        self.ValState.update()

    def generateFeatureBank(self):
        FeatureBank, LabelBank = [], []
        self.Model.eval()
        with torch.no_grad():
            for Batch in tqdm(self.MemDL, ncols=self.WinCols - 9, colour='magenta'):
                Batch = moveToDevice(Batch, self.opt.device)
                Img = Batch['image']
                Label = Batch['label']
                
                Feature = self.Model.forwardFeatures(Img)
                Feature = Feature.flatten(1)
                Feature = F.normalize(Feature, dim=1)
                FeatureBank.append(Feature)
                LabelBank.append(Label)
             
            # [D, N]
            self.FeatureBank = torch.cat(FeatureBank, dim=0).t().contiguous()
            # [N]
            self.LabelBank = torch.cat(LabelBank, dim=0)
 