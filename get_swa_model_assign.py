###python3 get_swa_model.py work_dirs/swin_tiny/ 8000 80000 work_dirs/swin_tiny/
import os
from argparse import ArgumentParser

import torch


def main():
    parser = ArgumentParser()
    parser.add_argument(
        'model_dir', help='the directory where checkpoints are saved')
    parser.add_argument(
        'pth_list',
        type=str,
        help='model_swa1.pth@model_swa2.pth@model_swa3.pth')
    parser.add_argument(
        'save_dir',
        type=str,
        help='the directory for saving the SWA model')
    args = parser.parse_args()

    model_dir = args.model_dir
    save_dir = args.save_dir
    pth_list = args.pth_list
    pth_list = pth_list.split("@")
    model_dirs = [
        os.path.join(model_dir, i)
        for i in pth_list
    ]
    print(model_dirs)
    models = [torch.load(model_dir) for model_dir in model_dirs]
    model_num = len(models)
    model_keys = models[-1]['state_dict'].keys()
    state_dict = models[-1]['state_dict']
    new_state_dict = state_dict.copy()
    ref_model = models[-1]

    for key in model_keys:
        sum_weight = 0.0
        for m in models:
            sum_weight += m['state_dict'][key]
        avg_weight = sum_weight / model_num
        new_state_dict[key] = avg_weight
    ref_model['state_dict'] = new_state_dict

    # save_dir = "/cache/swa_final.pth"
    torch.save(ref_model, save_dir)
    print('Model is saved at', save_dir)


if __name__ == '__main__':
    main()