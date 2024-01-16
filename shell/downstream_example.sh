# Classification
# STL10
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10 --num_repeat 3 --num_workers 4 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method common &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10 --num_repeat 3 --num_workers 4 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --sup_method common  --pretrained 2 --freeze_weight 3 --weight_name rotation_resnet50_wrmix5_30rr_v2_bshuffle_minr6_e200bs1024_Exp2.pth
## 5nn
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10 --num_repeat 3 --num_workers 4 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --clsval_mode 5nn --sup_method common &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname STL10 --num_repeat 3 --num_workers 4 --epochs 100 --save_point 80 --batch_size 1024 --lr 2e-4 --model_name resnet50 --optim adam --clsval_mode 5nn --sup_method common --pretrained 2 --freeze_weight 3 --weight_name rotation_resnet50_wrmix5_30rr_v2_bshuffle_minr6_e200bs1024_Exp2.pth

# setname: CIFAR10, CIFAR100
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname CIFAR10 --num_repeat 3 --num_workers 16 --epochs 100 --save_point 80 --batch_size 256 --lr 2e-4 --model_name resnet50 --sup_method common --optim adam &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname CIFAR10 --num_repeat 3 --num_workers 16 --epochs 100 --save_point 80 --batch_size 256 --lr 2e-4 --model_name resnet50 --sup_method common --optim adam --pretrained 2 --freeze_weight 3 --weight_name rotation_resnet50_wrmix5_30rr_v2_bshuffle_minr6_e200bs1024_Exp2.pth
## 5nn
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname CIFAR10 --num_repeat 3 --num_workers 16 --epochs 100 --save_point 80 --batch_size 256 --lr 2e-4 --model_name resnet50 --sup_method common --optim adam --clsval_mode 5nn &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname CIFAR10 --num_repeat 3 --num_workers 16 --epochs 100 --save_point 80 --batch_size 256 --lr 2e-4 --model_name resnet50 --sup_method common --optim adam --clsval_mode 5nn --pretrained 2 --freeze_weight 3 --weight_name rotation_resnet50_wrmix5_30rr_v2_bshuffle_minr6_e200bs1024_Exp2.pth

# Sports100
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname Sports100 --num_repeat 3 --num_workers 8 --epochs 100 --save_point 80 --batch_size 128 --lr 2e-4 --model_name resnet50 --sup_method common --optim adam &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname Sports100 --num_repeat 3 --num_workers 8 --epochs 100 --save_point 80 --batch_size 128 --lr 2e-4 --model_name resnet50 --sup_method common --optim adam --pretrained 2 --freeze_weight 3 --weight_name rotation_resnet50_wrmix5_30rr_v2_bshuffle_minr6_e200bs1024_Exp2.pth

# setname: PAD-UFES-20, Mammals
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname PAD2020S5 --num_split 5 --num_repeat 3 --num_workers 8 --epochs 100 --save_point 80 --batch_size 128 --lr 2e-4 --model_name resnet50 --optim adam --sup_method common &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname PAD2020S5 --num_split 5 --num_repeat 3 --num_workers 8 --epochs 100 --save_point 80 --batch_size 128 --lr 2e-4 --model_name resnet50 --optim adam --sup_method common --pretrained 2 --freeze_weight 3 --weight_name rotation_resnet50_wrmix5_30rr_v2_bshuffle_minr6_e200bs1024_Exp2.pth
## 5nn
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname PAD2020S5 --num_split 5 --num_repeat 3 --num_workers 8 --epochs 100 --save_point 80 --batch_size 128 --lr 2e-4 --model_name resnet50 --sup_method common --optim adam --clsval_mode 5nn &&
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --gpus 0 --setname PAD2020S5 --num_split 5 --num_repeat 3 --num_workers 8 --epochs 100 --save_point 80 --batch_size 128 --lr 2e-4 --model_name resnet50 --sup_method common --optim adam --clsval_mode 5nn --pretrained 2 --freeze_weight 3 --weight_name rotation_resnet50_wrmix5_30rr_v2_bshuffle_minr6_e200bs1024_Exp2.pth


# segmentation 
# setname: PASCALVOC, ISIC2018T1, FloodArea, TikTokDances
CUDA_VISIBLE_DEVICES=0 python main.py --task segmentation --gpus 0 --setname PASCALVOC --num_repeat 3 --num_workers 8 --epochs 100 --save_point 80 --batch_size 32 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adam --seg_head_name deeplabv3plus &&
CUDA_VISIBLE_DEVICES=0 python main.py --task segmentation --gpus 0 --setname PASCALVOC --num_repeat 3 --num_workers 8 --epochs 100 --save_point 80 --batch_size 32 --lr 1e-6 --warmup_init_lr 5e-5 --max_lr 5e-4 --model_name resnet50 --sup_method common --optim adam --seg_head_name deeplabv3plus --pretrained 2 --freeze_weight 3 --weight_name rotation_resnet50_wrmix5_30rr_v2_bshuffle_minr6_e200bs1024_Exp2.pth