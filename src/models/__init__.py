from .ResNet import *
from .ResNets import *
from .VGG import *
from .VGG_LTH import *

try:
    from .swin import *
except ModuleNotFoundError:
    swin_t = None

model_dict = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet20s": resnet20s,
    "resnet44s": resnet44s,
    "resnet56s": resnet56s,
    "vgg16_bn": vgg16_bn,
    "vgg16_bn_lth": vgg16_bn_lth,
}

if swin_t is not None:
    model_dict["swin_t"] = swin_t
