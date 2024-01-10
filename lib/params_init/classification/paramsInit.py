from ..singleInit import registerParams
from ..utils import pretextInit
from ...utils import CLASS_NAMES


@registerParams("classification")
def classficationParamsInit(opt):
    TestMode = not opt.setname
    
    # current situation
    opt.label_smoothing = 0.1
    opt.class_weights = True

    if not opt.class_names:
        SetName = opt.setname.lower()
        opt.class_names = CLASS_NAMES.get(SetName.replace('s%d' % opt.num_split, ''), None)
        if 'isic2018t1' in SetName:
            # isic2018t1 is not for classification
            opt.class_names = ['bkl', 'mel', 'nv']
        elif 'isic2019' in SetName:
            if 'mix' in SetName:
                opt.class_names = ['bcc', 'mel', 'nv'] # 'bkl'
        elif 'imagenet' in SetName:
            if 'annimal' in SetName:
                opt.class_names = ['mammal', 'bird', 'reptile'] # , 'fish'
                
        if opt.class_names is None:
            # cifar, inaturalist, imagenet
            opt.class_names = 'traversal'
        
    if not opt.metric_name:
        if 'common' in opt.sup_method:
            opt.metric_name = 'accuracy'
        else:
            opt.metric_name = 'loss'

    # self supervision
    ## general init
    opt = pretextInit(opt)
      
    # test mode
    if TestMode:
        if not SetName:
            opt.class_names = ['mammal', 'bird', 'reptile', 'test1', 'test2', 'test3'] 

        opt.knn_k = 2
        opt.epochs = 2
        opt.num_split = 2
        opt.num_supplement = 0
        opt.setname = 'sample/split'
        opt.resize_res = 224 # if not 'position' in opt.sup_method else 72
        
        opt.is_student = False
        opt.save_point = str(opt.epochs)
        opt.rot_degree = 30
    elif opt.epochs < 100 and 'common' in opt.sup_method:
        opt.num_repeat = 1

    if not opt.cls_loss_name:
        opt.cls_loss_name = opt.loss_name
        
    if 'focal' in opt.cls_loss_name:
        opt.label_smoothing = 0
      
    return opt