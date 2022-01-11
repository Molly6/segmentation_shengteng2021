# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import os
import cv2
import tqdm
# 容器内启动命令为python workspace/run.py /input_path /output_path
def main():
    parser = ArgumentParser()
    parser.add_argument('img_path', help='Image file')
    parser.add_argument('output', help='Checkpoint file')
    parser.add_argument('--config', default="/workspace/mmsegmentation-master/myconfigs/swin.py", help='Config file')
    parser.add_argument('--checkpoint', default="/workspace/mmsegmentation-master/checkpoints/latest.pth", help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test images
    # if os.path.exists(args.output):
    #     os.removedirs(args.output)
    for img in tqdm.tqdm(os.listdir(args.img_path)):
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        result = inference_segmentor(model, args.img_path + "/" + img)
        cv2.imwrite(args.output + "/" + img[:-4] + ".png", result[0])

if __name__ == '__main__':
    main()
