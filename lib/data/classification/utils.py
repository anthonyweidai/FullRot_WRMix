import random
from PIL import Image, ImageDraw

from ...utils import pair, pointCentreRotation


# position initailisation
def posInitWithRes(opt, SupRatio, GenMode, ImgSize=None):
    if ImgSize:
        W, H = ImgSize
    else:
        W, H = pair(opt.oriresize_res)
    
    CropRatio = opt.crop_ratio
    if not CropRatio:
        MinCropRatio = 0.8 if not GenMode else opt.mincrop_ratio
        MaxCropRatio = 1.0
        CropRatio = random.uniform(MinCropRatio, MaxCropRatio - SupRatio)
    else:
        if not opt.mincrop_ratio:
            MinCropRatio = 0.6
            
    RandomPos = opt.random_pos
    if CropRatio >= 1:
        CropRatio = 1
        RandomPos = False
    
    MinLen = min(W, H)
    CropLen = MinLen * CropRatio
     
    Left = (W - CropLen) / 2
    Upper = (H - CropLen) / 2
    Right = Left + CropLen
    Lower = Upper + CropLen
    SupLen = SupRatio * MinLen / 2 # Min using sublen prevent zero pixel near the boundary of image
    
    # Random rotation centre
    if RandomPos:
        # DeltaX = random.uniform(- W / 2 + CropLen / 2 + SupLen, W / 2 - CropLen / 2 - SupLen)
        # DeltaY = random.uniform(- H / 2 + CropLen / 2 + SupLen, H / 2 - CropLen / 2 - SupLen)
        
        if not opt.centre_rand:
            ## uniformly random
            DeltaX = random.uniform(- Left, Left)
            DeltaY = random.uniform(- Upper, Upper)
        else:
            ## random in the centre
            MaxOffset = max(MinLen * (CropRatio - MinCropRatio) / 2 - SupLen, 0) # - 0.1
            XMaxOffset = min(MaxOffset, Left)
            YMaxOffset = min(MaxOffset, Upper)
            DeltaX = random.uniform(- XMaxOffset, XMaxOffset)
            DeltaY = random.uniform(- YMaxOffset, YMaxOffset)
        
        Left = Left + DeltaX
        Upper = Upper + DeltaY
        Right = Right + DeltaX
        Lower = Lower + DeltaY
    
    return Left, Upper, Right, Lower, MinLen, CropLen, SupLen
    

def posInit(opt, ImgSize, SupRatio, GenMode):
    Left, Upper, Right, Lower, MinLen, CropLen, SupLen = \
        posInitWithRes(opt, SupRatio, GenMode, ImgSize)
    assert abs((Lower - Upper) - (Right - Left)) < 1e-5, "Not consistent height and width"
    # OutImg = imgCropResize(Img, opt.crop_resize)
    return Left, Upper, Right, Lower, MinLen, CropLen, SupLen


# rotation
def subCropafterRot(Img, RotSubRatio):
    # to prevent from damaged countour
    if RotSubRatio:
        W, H = Img.size
        SupLen = max(min(W, H) * RotSubRatio, 1)
        Img = Img.crop((SupLen, SupLen, W - SupLen, H - SupLen))
    return Img


def selfCentreRotation(Img: Image.Image, RotAngle, CropMode, RotSubRatio=0.01):
    # for cropRot and sliceImg, rotation around cropped image's centre
    RotImg = Img.rotate(RotAngle, Image.BILINEAR, expand=True) # counterclockwise, not inplace operation
    W, H = RotImg.size
    
    if CropMode == 1:
        return RotImg, [0, 0, W, H] # rotation will change centre
    else:
        # RotImg = subCropafterRot(RotImg, RotSubRatio) 
        WOri, HOri = Img.size
        SupLen = max(min(WOri, HOri) * RotSubRatio, 1)
        
        XOffset = (W - WOri) / 2
        YOffset = (H - HOri) / 2
        PosBox = [XOffset + SupLen, YOffset + SupLen, 
                  WOri + XOffset - SupLen, HOri + YOffset - SupLen]
        # PosBox = [round(x) for x in PosBox]
        
        Mask = Image.new("L", (W, H), 0)
        Draw = ImageDraw.Draw(Mask)
        Draw.ellipse(PosBox, fill=255)
        # Draw.pieslice([0, 0, W, H], 0, 360, fill=255)
        
        ImgNew = Image.new(RotImg.mode, (W, H)) # for black background
        ImgNew.paste(RotImg, (0, 0), Mask)
        return ImgNew.crop(PosBox), PosBox


def imgPointRotation(Img: Image.Image, RotAngle, PosBox, RotSubRatio=0.01):
    # for rotCrop, rotation around original image centre point
    RotImg = Img.rotate(RotAngle, Image.BILINEAR, expand=True) # counterclockwise, not inplace operation
    W, H = RotImg.size
    
    WOri, HOri = Img.size
    Diameter = PosBox[2] - PosBox[0]
    OriCentre = [(PosBox[2] + PosBox[0]) / 2, (PosBox[3] + PosBox[1]) / 2]
    
    # position box in the new coordinates
    XOffset = OriCentre[0] - WOri / 2
    YOffset = OriCentre[1] - HOri / 2
    
    # circle position after rotation, and crop box
    ImgCentre = [W / 2, H / 2]
    CircleCentre = [ImgCentre[0] + XOffset, ImgCentre[1] + YOffset]
    NewCentre = pointCentreRotation(ImgCentre, CircleCentre, - RotAngle)

    SupLen = max(min(WOri, HOri) * RotSubRatio, 1)
    CropPosBox = [
        NewCentre[0] - Diameter / 2 + SupLen,
        NewCentre[1] - Diameter / 2 + SupLen,
        NewCentre[0] + Diameter / 2 - SupLen,
        NewCentre[1] + Diameter / 2 - SupLen
        ] 
    
    # the new position box is in the extended rotated image
    Mask = Image.new("L", (W, H), 0)
    Draw = ImageDraw.Draw(Mask)
    Draw.ellipse(CropPosBox, fill=255)
    
    ImgNew = Image.new(RotImg.mode, (W, H)) # for black background
    ImgNew.paste(RotImg, (0, 0), Mask)
    
    # Mask = Image.new("L", (WOri, HOri), 0)
    # Draw = ImageDraw.Draw(Mask)
    # Draw.ellipse(PosBox, fill=255)
    
    # ImgNew2 = Image.new(Img.mode, (WOri, HOri)) # for black background
    # ImgNew2.paste(Img, (0, 0), Mask)
    return ImgNew.crop(CropPosBox)


def imgCropResize(Img, CropResize=False):
    W, H = Img.size
    if CropResize and W != H:
        MinLen = min(W, H)
        Left = (W - MinLen) / 2
        Upper = (H - MinLen) / 2
        # randomise the cropping area (not in the center)
        Right = random.uniform(0, Left) + MinLen
        Lower = random.uniform(0, Upper) + MinLen
        CropRegion = (Left, Upper, Right, Lower)
        return Img.crop(CropRegion)
    else:
        return Img


def cropRot(opt, Img: Image.Image, RotAngle, GenMode=True):
    SupRatio = 0
    Left, Upper, Right, Lower, _, CropLen, SupLen = posInit(opt, Img.size, SupRatio, GenMode)
    
    PosBox = [Left, Upper, Right, Lower]
    if opt.crop_mode == 3:
        CropRotImg = imgPointRotation(Img, RotAngle, PosBox)
    else:
        CropRotImg, PosBox = selfCentreRotation(Img.crop(PosBox), RotAngle, opt.crop_mode)
    
    if opt.remove_background:
        return CropRotImg
    else:
        WCrop, HCrop = CropRotImg.size
        
        XOffset = (CropLen - WCrop) / 2
        YOffset = (CropLen - HCrop) / 2
        PosBox = [round(Left + XOffset), round(Upper + YOffset)]
        
        ImgNew = Img.copy()
        ImgNew.paste(CropRotImg, PosBox)
        
        # Img.show()
        return ImgNew


def cropResizeScale(Img: Image.Image, ResizeRes, CropResize=False):
    if ResizeRes:
        W, H = Img.size
        if W != ResizeRes or H != ResizeRes:
            Img = imgCropResize(Img, CropResize)
            Img = Img.resize(pair(ResizeRes), Image.BICUBIC)
    return Img

