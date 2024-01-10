import argparse


class Opts(object):
    def __init__(self):
        self.Parser = argparse.ArgumentParser()
        # basic experiment setting
        self.Parser.add_argument('--task', default='classification',
                                help='classification | segmentation')
        self.Parser.add_argument('--sup_method', type=str, default='common',
                            help='common | rotation ')

        # system
        self.Parser.add_argument('--gpus', default='0', 
                                help='-1 for CPU, use comma for multiple gpus')
        self.Parser.add_argument('--num_workers', type=int, default=None,
                                help='dataloader threads. 0 for single-thread')


        # dataset
        self.Parser.add_argument('--custom_dataset_img_path', default='./dataset/',
                                help='custom dataset')
        # self.Parser.add_argument('--custom_dataset_ann_path', default='')
        self.Parser.add_argument('--setname', default='',
                                help='see lib/dataset/ for available datasets')
        self.Parser.add_argument('--num_split', type=int, default=1,
                                help='The number of cross-validation folders') # k-folds validation
        self.Parser.add_argument('--get_path_mode', type=int, default=None,
                                help='The mode of calling images from their path')
        self.Parser.add_argument('--num_repeat', type=int, default=None,
                                help='The number of repeated split') # k-folds validation
        self.Parser.add_argument('--resize_res', type=int, default=None,
                                help='The resize scale of transforms')
        self.Parser.add_argument('--class_names', type=list, default=None,
                                help='The list of class name')
        self.Parser.add_argument('--collate_fn_name', type=str, default=None,
                                help='The collate function name')
        self.Parser.add_argument('--drop_last', action='store_true',
                                help='Drop last in data loader, due to batch normlisation, \
                                the value will be different for the same dataset for \
                                    different batch size')
        self.Parser.add_argument('--loader_shuffle', action='store_false',
                                help='Shuffle data order while loading?')
        self.Parser.add_argument('--mask_fill', type=int, default=0,
                                help='Fill the mask with background class label')
        self.Parser.add_argument('--random_aug_order', action='store_true', 
                            help='Use random order for augmentation methods?')
        self.Parser.add_argument('--mask_ratio', type=float, default=None,
                                help='The mask ratio of masking image in data loading.')
        self.Parser.add_argument('--use_meanstd', action='store_false',
                                help='Use mean std to process the image RGB channels?')


        # task
        self.Parser.add_argument('--num_classes', type=int, default=None,
                                help='The number of classes')
        self.Parser.add_argument('--init_weight', action='store_false', 
                                help='Apply weight initialisation?')
        ## classification
        self.Parser.add_argument('--model_name', default='mobilenetv3_small', 
                                help='model architecture')
        self.Parser.add_argument('--cls_num_classes', type=int, default=None,
                                help='The number of classes')


        ## segmentation
        self.Parser.add_argument('--seg_num_classes', type=int, default=None,
                                help='The number of classes')
        self.Parser.add_argument('--seg_model_name', type=str, default="encoder_decoder",
                                help='Segmentation models\' name')
        '''
        "encoder_decoder"
        '''                         
        self.Parser.add_argument('--seg_head_name', type=str, default="deeplabv3",
                                help='Segmentation heads\' name')
        '''
        "deeplabv3", "deeplabv3plus",
        '''
        self.Parser.add_argument('--use_sep_conv', action='store_true',
                                help='True | False') # deeplabv3
        
        # train
        self.Parser.add_argument('--epochs', type=int, default=70,
                                help='total training epochs.')
        self.Parser.add_argument('--batch_size', type=int, default=4,
                                help='batch size') # 8 multiple
        self.Parser.add_argument('--stop_station', type=int, default=100,
                                help='The stop epoch NO. for efficient training')
        self.Parser.add_argument('--num_supplement', type=int, default=0,
                                help='The number of supplement exp for cross validation')
        self.Parser.add_argument('--exp_type', type=str, default='classification',
                                help='The name of exp folder (task)') 
        self.Parser.add_argument('--exp_level', type=str, default='',
                                help='The focus comparison name for a new level of folder') 
        self.Parser.add_argument('--target_exp', type=int, default=None,
                                help='The target exp folder location for supplement')
        self.Parser.add_argument('--dest_path', type=str, default=None,
                                help='The full target exp folder path')
        self.Parser.add_argument('--save_point', type=str, default='80', # 30,60,90
                                help='when to save the model to disk.')
        self.Parser.add_argument('--clsval_mode', type=str, default='linear',
                                help='linear | 5nn') # only in classification task
        self.Parser.add_argument('--knn_k', type=int, default=5,
                                help='The numer of nearest neighbor in kNN monitor')
        self.Parser.add_argument('--val_start_epoch', type=int, default=None,
                                help='The validation starting point')

        # optim
        self.Parser.add_argument('--optim', default='adam')
        '''
        OptimChoices = ['adam', 'adamw', 'adamax', 'sgd', 'asgd',
                    'rmsprop', 'rprop', 'lbfgs', 'adadelta', 'adagrad',
                    'lars']
        '''
        self.Parser.add_argument('--lr', type=float, default=None, 
                            help='The minimum learning rate')
        self.Parser.add_argument('--warmup_init_lr', type=float, default=None, 
                                help='warming up learning rate for schedular')
        self.Parser.add_argument('--max_lr', type=float, default=None, 
                                help='maximum learning rate for schedular')
        self.Parser.add_argument('--lr_decay', action='store_false',
                                help='learning rate decay')
        self.Parser.add_argument('--schedular', type=str, default='mycosine',
                                help='mycosine | base')
        self.Parser.add_argument('--weight_decay', type=float, default=0,
                                help='mycosine')
        self.Parser.add_argument('--milestones', type=int, default=None,
                                help='milestones for learning rate decay')
        ## adam
        self.Parser.add_argument('--beta1', default=0.9)
        self.Parser.add_argument('--beta2', default=0.999)
        self.Parser.add_argument('--amsgrad', action='store_true',
                                help='whether to use the AMSGrad variant of \
                                this algorithm from the paper \
                                    `On the Convergence of Adam and Beyond`_(default: False)')
        ## sgd
        self.Parser.add_argument('--nesterov', action='store_true',
                                help='enables Nesterov momentum (default: False)')
        self.Parser.add_argument('--momentum', type=float, default=0,
                                help='momentum factor (default: 0)')
        ## lars
        self.Parser.add_argument('--eta', type=float, default=1e-3,
                                help='LARS coefficient as used in the paper (default: 1e-3)')
        self.Parser.add_argument('--dampening', type=float, default=0,
                                help='dampening for momentum (default: 0)')


        # transfer
        self.Parser.add_argument('--load_model_path', default='./savemodel/',
                                help='path to pretrained model')
        self.Parser.add_argument('--pretrained', type=int, default=0, 
                                help='Use transfer learning?')
        self.Parser.add_argument('--freeze_weight', type=int, default=0, 
                                help='Freezing for partially transfer learning warm-up')
        self.Parser.add_argument('--weight_name', type=str, default=None, 
                                help='for partially transfer learning warm-up')


        # metrics
        self.Parser.add_argument('--metric_name', type=str, default=None,
                                help='loss | accuracy | iou')
        self.Parser.add_argument('--loss_coeff', type=list, default=None) # [0.3, 0.3, 1]
        self.Parser.add_argument('--loss_name', type=str, default='cross_entropy')
        '''
        cross_entropy, focal,
        mdca_cross_entropy, mdca_focal,
        poly_cross_entropy, poly_focal,
        '''
        self.Parser.add_argument('--cls_loss_name', type=str, default=None)
        self.Parser.add_argument('--seg_loss_name', type=str, default=None)
        self.Parser.add_argument('--class_weights', action='store_true',
                                help='Use class sensitive loss?')
        self.Parser.add_argument('--label_smoothing', type=float, default=0, # 0.1
                                help='The label smoothing params for cross entropy.')
        self.Parser.add_argument('--loss_reduction', type=str, default='mean',
                                help='mean | sum')
        self.Parser.add_argument('--use_aux_head', action='store_true',
                                help='Use supplementary head for segmentation?')
        self.Parser.add_argument('--aux_weight', type=float, default=0.4,
                                help='The loss weight of segmentation auxiliary branch.')
        self.Parser.add_argument('--metric_type', type=str, default='micro',
                                help='macro | micro')
        self.Parser.add_argument('--ignore_idx', type=int, default=-100,
                                help='ignore background in segmentation loss calculation')

        ## supplementary metrics for classification
        self.Parser.add_argument('--sup_metrics', action='store_true',
                                help='For small dataset. Supplementary metrics for classifcation, \
                                including recall, precision, specificity, F1Score')
        self.Parser.add_argument('--topk', type=tuple, default=(1, 5),
                                help='For small dataset. Supplementary metrics for classifcation, \
                                including recall, precision, specificity, F1Score')

        # self supervision
        ## general
        self.Parser.add_argument('--selfsup_valid', action='store_true',
                                help='Use validation for self supervision?')
        self.Parser.add_argument('--views', type=int, default=2,
                                help='The number of novel views (output images) for self supervision')
        self.Parser.add_argument('--crop_mode', type=int, default=None,
                                help='0 no crop (original rot) | 1 rectangle crop | 2 circle crop \
                                    | 3 circle crop and rotation around original image point  ')
        self.Parser.add_argument('--crop_ratio', type=float, default=None,
                                help='How large ratio of region do you want to crop? (0, 1)')
        self.Parser.add_argument('--mincrop_ratio', type=float, default=0.6,
                                help='How minimum ratio of region do you want to crop? (0, 1)')
        self.Parser.add_argument('--random_pos', action='store_false',
                                help='Do you use random position of rotated region?')
        self.Parser.add_argument('--centre_rand', action='store_true',
                                help='Random position in the centre or uniform pos?')
        self.Parser.add_argument('--crop_resize', action='store_false', # If false, could add additional len info (include swim mode)
                                help='Do you want to resize image before croping?')
        self.Parser.add_argument('--sup_path', type=str, default='',
                                help='The root path of pretext saved files')
        self.Parser.add_argument('--pretext_format', type=str, default=None,
                                help='.bmp | .jpg') # '.bmp' bmp is super large 80 x memory storage!!!
        self.Parser.add_argument('--remove_background', action='store_false',
                                help='Do you want to remove background after cropping?')
        self.Parser.add_argument('--batch_shuffle', action='store_true',
                                help='Shuffle image on each batch for self-supervision?')
        self.Parser.add_argument('--extend_val', action='store_false',
                                help='Extending validation set for self-supervised training?')

        ## rotation
        self.Parser.add_argument('--rot_degree', type=int, default=None,
                                help='The degree of angle of rotation self-sup') # 0 < degree <= 120
        self.Parser.add_argument('--region_rot', action='store_false',
                                help='Rotate only center circular region? True | False')

        ## mix
        self.Parser.add_argument('--mix_method', type=str, default=None,
                                help='mixup | cutmix | hmix | gmix | wrmix | hybrid')
        self.Parser.add_argument('--mix_alpha', type=float, default=1.,
                                help='The alpha value for lambda beta distribution.')
        self.Parser.add_argument('--hmix_ratio', type=float, default=0.75,
                                help='The value of hmix ratio')
        self.Parser.add_argument('--mix_gamma', type=float, default=0.5, # 0.3 for rotmix
                                help='How large possibility will you use mixup')
        self.Parser.add_argument('--mix_kappa', type=float, default=1.,
                                help='How large possibility will you use intra and inter image mixup')
        self.Parser.add_argument('--mix_base_bias', type=float, default=0.2,
                                help='The base bias factor for mixup lambda')
        self.Parser.add_argument('--mix_mincrop_ratio', type=float, default=0.5,
                                help='The base bias factor for mixup lambda')
        self.Parser.add_argument('--mix_maskrot', action='store_false',
                                help='Rotate mixup mask?')
     
    def parse(self, args=''):
        if args == '':
            opt = self.Parser.parse_args()
        else:
            opt = self.Parser.parse_args(args)
        return opt