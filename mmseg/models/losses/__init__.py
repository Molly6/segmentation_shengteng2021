# Copyright (c) OpenMMLab. All rights reserved.
from .accuracy import Accuracy, accuracy
from .cross_entropy_loss import (CrossEntropyLoss, LabelSmoothingCrossEntropy, FocalLoss2d, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy, SoftCrossEntropyLoss)
from .dice_loss import DiceLoss
from .lovasz_loss import LovaszLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss

__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy', 'LabelSmoothingCrossEntropy', 'FocalLoss2d',
    'mask_cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
    'weight_reduce_loss', 'weighted_loss', 'LovaszLoss', 'DiceLoss', 'SoftCrossEntropyLoss'
]
