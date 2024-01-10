import os
import sys

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer, default_argument_parser, default_setup, launch
)
from detectron2.evaluation import (COCOEvaluator, PascalVOCDetectionEvaluator, 
                                   CityscapesInstanceEvaluator)

'''
Modified https://github.com/facebookresearch/moco/tree/main/detection
Available datasets: 
voc_2012_trainval, voc_2007_trainval, voc_2007_test
'''


class TrainEngine(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if "coco" in dataset_name:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif "voc" in dataset_name:
            return PascalVOCDetectionEvaluator(dataset_name)
        elif "city" in dataset_name:
            return CityscapesInstanceEvaluator(dataset_name)


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        Model = TrainEngine.build_model(cfg)
        DetectionCheckpointer(Model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        Res = TrainEngine.test(cfg, Model)
        return Res

    Trainer = TrainEngine(cfg)
    Trainer.resume_or_load(resume=args.resume)

    return Trainer.train()


if __name__ == "__main__":
    # sys.argv[1]
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    