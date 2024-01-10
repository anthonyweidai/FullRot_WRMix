
## Transferring to Detection

The `trainDet.py` script reproduces the object detection experiments on Pascal VOC.

### Instruction

1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

1. Convert a pre-trained model to detectron2's format:
   ```
   python3 tools/convertPretrain2D2.py input.pth.tar output.pkl
   ```

1. Put dataset under "./datasets" directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by detectron2.

1. Run training:
   ```
   python trainDet.py --config-file lib/detection/configs/faster_rcnn_R_50_FPN.yaml \
	--num-gpus 1 MODEL.WEIGHTS ./output.pkl
   ```

***Note:*** These results are means of 3 trials.