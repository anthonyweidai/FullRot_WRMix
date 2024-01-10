import numpy as np
from tqdm import tqdm
from pathlib import Path
from imageio import imsave

# STL10
'''reference: https://www.kaggle.com/code/pratt3000/generate-stl10/notebook'''
HomePath = '.'

MetaNames = ['train', 'test', 'unlabelled']
for MetaName in MetaNames:
    DestPath = '%s/%s' % (HomePath, MetaName)
    
    # read images
    with open('%s/%s_X.bin' % (HomePath, MetaName), 'rb') as f:
        Metadata = np.fromfile(f, dtype=np.uint8)

        Images = np.reshape(Metadata, (-1, 3, 96, 96))
        Images = np.transpose(Images, (0, 3, 2, 1))
    
    # read labels
    if 'unlabelled' not in MetaName:
        with open('%s/%s_y.bin' % (HomePath, MetaName), 'rb') as f:
            Labels = np.fromfile(f, dtype=np.uint8)
    else:
        Labels = [1] * len(Images)
    NumClasses = set(Labels).__len__()
    
    # writing image with label
    Counter = np.zeros((NumClasses, 1), dtype=np.int32)
    ClassNames = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
    for i, Img in enumerate(tqdm(Images, colour='green', ncols=60)):
        Label = Labels[i] - 1
        Counter[Label] += 1
        
        if NumClasses > 1:
            SaveStr = '%s_%d' % (ClassNames[Label], Counter[Label])
        else:
            SaveStr = 'unlabelled_img_%d' % (Counter[Label]) 
            
        Path(DestPath).mkdir(parents=True, exist_ok=True)
        imsave('%s/%s.png' %(DestPath, SaveStr), Img, format="png")