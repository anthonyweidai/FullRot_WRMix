from glob import glob
from pathlib import Path    

from lib.detection.utils import convertPretrain2Detectron2


if __name__ == "__main__":
    SingleMode = 1
    if SingleMode:
        ModelName = 'rotation_wrmix_30rr_v2_bshuffle_minr6_e200bs1024_Exp4.pth' # sys.argv[1]
        OutName = Path(ModelName).stem + '.pkl' # sys.argv[2]

        convertPretrain2Detectron2(ModelName, OutName)
    else:
        SetName = 'STL10_unlabelled'
        WeightFolder = './savemodel/%s/' % SetName
        DestFolder = '%s/pkl' % WeightFolder
        Path(DestFolder).mkdir(parents=True, exist_ok=True)
        
        for WeightPath in glob("%s*.pth" % (WeightFolder)):
            OutName = '%s/%s%s' % (DestFolder, Path(WeightPath).stem, '.pkl') # sys.argv[2]
        
            convertPretrain2Detectron2(WeightPath, OutName)