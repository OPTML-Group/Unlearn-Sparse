from models.ResNet import *
from models.ResNets import *
from models.VGG import * 
from models.VGG_LTH import *

model_dict = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet20s': resnet20s,
    'resnet44s': resnet44s,
    'resnet56s': resnet56s,
    'vgg16_bn': vgg16_bn,
    'vgg16_bn_lth': vgg16_bn_lth,
}