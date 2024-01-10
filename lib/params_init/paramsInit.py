import os
import math

from .singleInit import singleInit
from .utils import initPathMode
from ..utils import RESDICT, getOnlyFolderNames, getFilePathsFromSubFolders


def paramsInit(opt):
    # single init
    opt = singleInit(opt)
    SetName = opt.setname.lower()
  
  
    '''Init after single task init'''
    # device
    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    
    # dataset
    ## path
    if not os.path.isdir(opt.custom_dataset_img_path):
        opt.custom_dataset_img_path = '../dataset/'
    
    opt.dataset_path = opt.custom_dataset_img_path + opt.setname
        
    ## resize resolution
    if 'resize' in SetName:
            DefaultResizeRes = RESDICT.get(SetName.replace('_resize', ''), 
                                         RESDICT['default'][opt.task])
    else:
        DefaultResizeRes = RESDICT.get(SetName, RESDICT['default'][opt.task])
     
    if opt.resize_res is None:
        opt.resize_res = DefaultResizeRes

    ## get image path mode
    if not opt.get_path_mode:
        opt.get_path_mode = initPathMode(opt.dataset_path, SetName)
        
    ## class name
    if 'traversal' in opt.class_names:
        opt.class_names = getOnlyFolderNames(opt.dataset_path + '/train')
        
    ## collate function name
    if not opt.collate_fn_name:
        if 'common' not in opt.sup_method:
            if opt.batch_shuffle:
                opt.collate_fn_name = "pretext_batch_shuffle"
            else:
                opt.collate_fn_name = "pretext"
        else:
            opt.collate_fn_name = "default"
            
    
    # train
    ## model
    if not os.path.isdir(opt.load_model_path):
        opt.load_model_path = '../savemodel/'
    
    opt.model_name = opt.model_name.lower()
        
    opt.save_point = [int(i) for i in opt.save_point.split(',')]
    opt.save_point.sort()
    
    if not opt.num_repeat:
        opt.num_repeat = opt.num_split
    
    ## exp level
    if opt.exp_level == '':
        if 'common' in opt.sup_method:
            # replace for sample dataset
            opt.exp_level = opt.setname.lower().replace('/', '_') 
    
    ## start epoch
    if opt.val_start_epoch is None:
        opt.val_start_epoch = 0
  
    ## optimiser
    if 'cosine' in opt.schedular:
        opt.stop_station = opt.epochs
        if opt.lr is None:
            if 'sgd' in opt.optim:
                opt.lr = 0
                opt.max_lr = 0.05 * opt.batch_size / 256
            else:
                opt.lr = 2e-4
        
        if opt.max_lr is None:
            opt.max_lr = opt.lr * 10
        
        if opt.warmup_init_lr is None:
            opt.warmup_init_lr = opt.lr
    else:
        if opt.lr is None:
            opt.lr = 0.002
    
    if not opt.lr:
        if 'sgd' in opt.optim:
            opt.lr = 1e-7
            opt.max_lr = opt.lr_factor * opt.batch_size / 256
        else:
            opt.lr = 2e-4
            opt.max_lr = 2e-3
    elif not opt.max_lr:
        opt.max_lr = opt.lr * 10
    
    if opt.milestones is None:
        opt.milestones = math.ceil(0.1 * opt.epochs)
  
    # if opt.is_student:
    #   opt.freeze_weight = 0 if not opt.is_distillation else 0 # 0, 1, 2, 3
    
    if 'adamw' in opt.optim:
        if opt.weight_decay is None:
            opt.weight_decay = 1.e-2
  
    ## pretrained weight
    if opt.weight_name:
        WeightPool = getFilePathsFromSubFolders(opt.load_model_path)
        opt.pretrained_weight = [WeightPath for WeightPath in WeightPool \
            if opt.weight_name in WeightPath][0]
    else:
        opt.pretrained_weight = None
        
    if 'ml_' in opt.model_name:
        opt.loss_mode = 1

    ## number of views
    if 'common' in opt.sup_method:
        opt.views = 1
  
    # number of classes
    if 'common' in opt.sup_method:
        opt.num_classes = len(opt.class_names)
        opt.seg_num_classes = opt.num_classes
        opt.det_num_classes = opt.num_classes
    else:
        if not opt.num_classes:
            if 'rotation' in opt.sup_method:
                opt.num_classes = round(360 / opt.rot_degree)
                assert opt.views <= opt.num_classes, \
                    "Got %d output rotated image which is larger than %d classes" \
                        % (opt.views, opt.num_classes)
                
        elif not opt.rot_degree:
            opt.rot_degree = round(360 / opt.num_classes)
      
    if not opt.cls_num_classes:
        opt.cls_num_classes = opt.num_classes
    
    # metrics
    if 'common' not in opt.sup_method and not opt.selfsup_valid:
        opt.metric_name = 'loss'
        
    if opt.sup_metrics or opt.cls_num_classes < 5:
        opt.topk = (1, )
        opt.sup_metrics = True
        
    return opt