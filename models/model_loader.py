import torch.nn as nn

import models.alexnet as alexnet
import models.resnet50 as resnet50
import models.vgg16 as vgg16


def load_model(arch, code_length):
    """
    Load CNN model.

    Args
        arch(str): CNN model name.
        code_length(int): Hash code length.

    Returns
        model(torch.nn.Module): CNN model.
    """
    if arch == 'alexnet':
        model = alexnet.load_model('alexnet', code_length)
    elif arch == 'vgg16':
        model = vgg16.load_model(code_length)
    elif arch == 'resnet50':
        model = resnet50.load_model(code_length)
    else:
        raise ValueError('Invalid cnn model name!')

    return model

