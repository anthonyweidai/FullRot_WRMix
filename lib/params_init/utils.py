from ..utils import makeDivisible, getOnlyFolderNames


def pretextInit(opt):
    if opt.rot_degree:
        assert opt.rot_degree > 0 and opt.rot_degree <= 120, \
            "Got {} degree, which is not in range (0, 120}".format(opt.rot_degree)
    if opt.rot_degree != 90:
        opt.region_rot = True # prevent the image angle being the supervised info
        
    if opt.crop_mode is None:
        if opt.region_rot and 'rot' in opt.sup_method:
            opt.crop_mode = 2
    elif opt.crop_mode == 0:
            opt.region_rot = False
    elif opt.crop_mode == 1 or opt.crop_mode == 3:
        # aleady rotate with expand
        opt.remove_background = True # logical lock
        if not opt.region_rot: # 'rot' in opt.sup_method
            opt.rot_degree = 90
    
    if opt.crop_ratio == 1:
        opt.random_pos = False

    return opt


def initPathMode(DatasetPath, SetName):
    # for param initialisation and norm counter
    FolderNames = getOnlyFolderNames(DatasetPath)
    if 'split' in FolderNames or 'split1' in FolderNames:
        PathMode = 1
    elif 'cifar' in SetName:
        PathMode = 3
    else:
        PathMode = 4
    return PathMode
