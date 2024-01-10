import pickle as pkl

import torch


def convertPretrain2Detectron2(ModelName, OutName):
    '''Modified version of 
    https://github.com/facebookresearch/detectron2/blob/
    32570b767e1b69516c58734fe6cc46005bab2aae/tools/convert-torchvision-to-d2.py
    '''
    print("Converting %s to its pkl version" % ModelName)
    Model = torch.load(ModelName, map_location="cpu")

    NewModel = {}
    for k in list(Model.keys()):
        Key = k.lower()
        if any(s in Key for s in ['classifier', 'fc', 'encoderk']):
            continue
        
        OldKey = Key
        if "layer" not in Key:
            Key = "stem." + Key
        for t in [1, 2, 3, 4]:
            Key = Key.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3, 4, 5]:
            Key = Key.replace("conv{}.0".format(t), "conv{}".format(t))
        for t in [1, 2, 3]:
            Key = Key.replace("conv{}.bn".format(t), "conv{}.norm".format(t))
            # Key = Key.replace("conv{}.0.bn".format(t), "conv{}.norm".format(t))   
        Key = Key.replace("encoder.", "") # for barlow
        Key = Key.replace("encoderq.", "") # for moco
        Key = Key.replace("downsample.conv", "shortcut")
        Key = Key.replace("downsample.bn", "shortcut.norm")
        Key = Key.replace("conv.weight", "weight")
        # print(OldKey, "->", Key)
        NewModel[Key] = Model.pop(k).detach().numpy()

    Res = {"model": NewModel, "__author__": "Anthony", "matching_heuristics": True}

    with open(OutName, "wb") as f:
        pkl.dump(Res, f)
        
    if Model:
        print("Unconverted keys:", Model.keys())