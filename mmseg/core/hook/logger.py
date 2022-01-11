import logging
import os
import sys
from collections import Counter
from typing import Tuple

import mmcv
import numpy as np
import torch
from mmcv.runner.dist_utils import get_dist_info
from mmcv.utils import get_logger

try:
    import wandb
except:
    wandb = None

_log_counter = Counter()


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name="mmseg.ssod", log_file=log_file, log_level=log_level)
    logger.propagate = False
    return logger


def _find_caller():
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = r"ssod"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


def convert_box(tag, boxes, box_labels, class_labels, std, scores=None):
    if isinstance(std, int):
        std = [std, std]
    if len(std) != 4:
        std = std[::-1] * 2
    std = boxes.new_tensor(std).reshape(1, 4)
    wandb_box = {}
    boxes = boxes / std
    boxes = boxes.detach().cpu().numpy().tolist()
    box_labels = box_labels.detach().cpu().numpy().tolist()
    class_labels = {k: class_labels[k] for k in range(len(class_labels))}
    wandb_box["class_labels"] = class_labels
    assert len(boxes) == len(box_labels)
    if scores is not None:
        scores = scores.detach().cpu().numpy().tolist()
        box_data = [
            dict(
                position=dict(minX=box[0], minY=box[1], maxX=box[2], maxY=box[3]),
                class_id=label,
                scores=dict(cls=scores[i]),
            )
            for i, (box, label) in enumerate(zip(boxes, box_labels))
        ]
    else:
        box_data = [
            dict(
                position=dict(minX=box[0], minY=box[1], maxX=box[2], maxY=box[3]),
                class_id=label,
            )
            for i, (box, label) in enumerate(zip(boxes, box_labels))
        ]

    wandb_box["box_data"] = box_data
    return {tag: wandb.data_types.BoundingBoxes2D(wandb_box, tag)}


def color_transform(img_tensor, mean, std, to_rgb=False):
    img_np = img_tensor.detach().cpu().numpy().transpose((1, 2, 0)).astype(np.float32)
    return mmcv.imdenormalize(img_np, mean, std, to_bgr=not to_rgb)


def log_every_n(msg: str, n: int = 50, level: int = logging.DEBUG, backend="auto"):
    """
    Args:
        msg (Any):
        n (int):
        level (int):
        name (str):
    """
    caller_module, key = _find_caller()
    _log_counter[key] += 1
    if n == 1 or _log_counter[key] % n == 1:
        if isinstance(msg, dict) and (wandb is not None) and (wandb.run is not None):
            wandb.log(msg, commit=False)
        else:
            get_root_logger().log(level, msg)
