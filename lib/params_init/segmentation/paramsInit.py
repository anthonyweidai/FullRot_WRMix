from ..singleInit import registerParams
from ...utils import CLASS_NAMES


@registerParams("segmentation")
def segmentationParamsInit(opt):
    TestMode = not opt.setname
    
    if not opt.class_names:
        opt.num_split = 1
        opt.mask_fill = 255
        opt.ignore_idx = 255

        SetName = opt.setname.lower()
        opt.class_names = ["background"]
        if SetName:
            # for test mode
            opt.class_names.extend(CLASS_NAMES[SetName.lower()])
    
    if not opt.metric_name:
      opt.metric_name = 'iou'
    
    # test mode
    if TestMode:
        if not SetName:
            opt.class_names.extend(CLASS_NAMES['pascalanimal']) # "person", 
            
        if 'common' in opt.sup_method:
            opt.setname = 'sample_seg'

        opt.epochs = 2
        opt.num_supplement = 0
        opt.target_exp = None
        opt.is_student = False
    elif opt.epochs < 100 and 'common' in opt.sup_method:
        opt.num_repeat = 1
    
    if not opt.seg_loss_name:
      opt.seg_loss_name = opt.loss_name
    
    return opt