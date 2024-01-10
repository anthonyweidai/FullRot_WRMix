# System
TextColors = {
    'end_colour': '\033[0m',
    'bold': '\033[1m', # 033 is the escape code and 1 is the color code 
    'error': '\033[31m', # red
    'light_green': '\033[32m',
    'light_yellow': '\033[33m',
    'light_blue': '\033[34m',
    'light_cyan': '\033[36m',
    'warning': '\033[37m', # white
}


# Dataset
RESDICT = {
    'default': {'classification' : 224, 'segmentation': 512, 
                       'detection': 320, 'ins_segmentation': 512}, # 'regression'
    **dict.fromkeys(['cifar10', 'cifar100'], 32),
    'ade20k': 512,
    'pascalvoc': 512,
    'isic2019': 224, 'isic2018t1': 512, 
    'pad2020': 224,
    **dict.fromkeys(['stl10', 'stl10_unlabelled'], 96),
}


CLASS_NAMES = {
    'pad2020': ['ack', 'bcc', 'bkl', 'mel', 'nv', 'scc'],
    'isic2018t1': ['symptoms'],
    'cifar': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'pascalvoc': [
        "aeroplane", "bicycle", "bird", "boat", "bottle", 
        "bus", "car", "cat", "chair", "cow", "diningtable", 
        "dog", "horse", "motorbike", "person", "potted plant", "sheep",
        "sofa", "train", "tv monitor"
        ],
    'stl10': ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"],
    'ade20k': [
        "wall", "building", "sky", "floor", "tree", "ceiling", "road", 
        "bed ", "windowpane", "grass", "cabinet", "sidewalk", "person", 
        "earth", "door", "table", "mountain", "plant", "curtain", "chair", 
        "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", 
        "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", 
        "lamp", "bathtub", "railing", "cushion", "base", "box", "column", "signboard", 
        "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator", 
        "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway", 
        "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", 
        "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", 
        "arcade machine", "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", 
        "streetlight", "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", 
        "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", 
        "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", 
        "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", 
        "ball", "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", 
        "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase", 
        "traffic light", "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor", 
        "bulletin board", "shower", "radiator", "glass", "clock", "flag",
    ],
}