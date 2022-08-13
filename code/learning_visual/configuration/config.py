import logging
import os
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from data_layer.channels import Channels
from util.files_operations import make_folder

print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION')
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
# call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
# print('Active CUDA Device: GPU', torch.cuda.current_device())
# print('Available devices ', torch.cuda.device_count())
# print('Current cuda device ', torch.cuda.current_device())
use_cuda = torch.cuda.is_available()
print("USE CUDA=" + str(use_cuda))
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


def parse_args(model, target_channel, exp_num=None):
    parser = ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', choices=('train', 'predict'))

    DATA_DIR, METADATA_PATH, LOG_DIR, IMAGES_PATH, EXP_DIR = get_paths(exp_num, model.name, target_channel)
    parser.add_argument('--data_path', type=Path, default=DATA_DIR,
                        help='path to the data root. It assumes format like in Kaggle with unpacked archives')
    parser.add_argument('--metadata_path', type=Path, default=METADATA_PATH,
                        help='path to the data root. It assumes format like in Kaggle with unpacked archives')
    parser.add_argument('--images_path', type=Path, default=IMAGES_PATH,
                        help='path to the data root. It assumes format like in Kaggle with unpacked archives')
    parser.add_argument('--log_dir', type=Path, default=LOG_DIR,
                        help='path to experiment logs.')
    parser.add_argument('--exp_dir', type=Path, default=EXP_DIR,
                        help='path to experiment results.')

    parser.add_argument('--plates_split', type=list,
                        default=[[24509, 24633, 24792, 25912], [24294, 24311, 25938, 25985, 25987]],
                        help='plates split between train and test. left is train and right is test')
    parser.add_argument('--test_samples_per_plate', type=int, default=-1,
                        help='number of test samples for each plate. if None, all plates are taken')

    parser.add_argument('--split-ratio', type=float, default=0.8,
                        help='split ratio between train and validation. value in [0,1]')

    parser.add_argument('--num-data-workers', type=int, default=6,
                        help='number of data loader workers')
    parser.add_argument('--device', type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help='device for running code')
    parser.add_argument('--seed', type=int, default=42,
                        help='global seed (for weight initialization, data sampling, etc.). '
                             'If not specified it will be randomized (and printed on the log)')

    parser.add_argument('--target_channel', type=str, default=target_channel.name, choices=('AGP', 'DNA', 'ER', 'Mito', 'RNA'),
                        help='the channel predicted by the network')
    parser.add_argument('--num_input_channels', type=int, default=model.params['n_input_channels'], choices=(1, 4, 5),
                        help='defines what autoencoder is trained (4to1, 1to1, 5to5)')
    parser.add_argument('--input_size', type=tuple, default=model.params['input_size'],
                        help='width and height input into the network')

    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--epochs', type=int, default=model.params['epochs'])
    parser.add_argument('--lr', type=float, default=model.params['lr'])
    parser.add_argument('--minimize_net_factor', type=int, default=model.params['minimize_net_factor'],
                        help='reduces the network number of convolution maps by a factor')

    parser.add_argument('--checkpoint', type=str,
                        default='',
                        help='path to load existing model from')

    args = parser.parse_known_args()[0]

    if args.seed is None:
        args.seed = random.randint(0, 10 ** 9)

    args.target_channel = eval(f'Channels.{args.target_channel}')

    setup_logging(args)
    setup_determinism(args)

    return args


def get_paths(exp_num=None, model_type='UNET4TO1', target_channel=Channels.AGP):
    from datetime import datetime
    ROOT_DIR = '/storage/users/g-and-n/visual_models_results/'

    DATA_DIR = f"/storage/users/g-and-n/plates"

    EXP_DIR = f"{ROOT_DIR}"
    # if exp_num is None:
    #     exp_num = get_exp_num(EXP_DIR)
    EXP_DIR = os.path.join(EXP_DIR, str(exp_num), model_type, "channel " + target_channel.name)
    # EXP_DIR = os.path.join(EXP_DIR, str(exp_num), "channel " + str(target_channel))
    # exp_num = get_exp_num(EXP_DIR)
    if exp_num is not None:
        make_folder(EXP_DIR)
    METADATA_PATH = os.path.join(DATA_DIR, 'metadata_predict')
    IMAGES_PATH = os.path.join(DATA_DIR, 'images')

    LOG_DIR = f"{EXP_DIR}"

    return DATA_DIR, METADATA_PATH, LOG_DIR, IMAGES_PATH, EXP_DIR


def setup_logging(args):
    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr)]
    # if args.mode == 'train':
    #     handlers.append(logging.FileHandler(args.save + '.log', mode='w'))
    # if args.mode == 'predict':
    #     handlers.append(logging.FileHandler(args.load + '.output.log', mode='w'))
    logging.basicConfig(level=logging.INFO, format=head, style='{', handlers=handlers)
    logging.info('Start with arguments {}'.format(args))


def setup_determinism(args):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def get_checkpoint(LOG_DIR, model_name, target_channel):
    base = f'{LOG_DIR}/log_dir/'
    try:
        ver_dir = os.listdir(base)
        if ver_dir:
            ver_dir = ver_dir[0]
            chk_dir = os.path.join(base, ver_dir, 'checkpoints')
            chk_file = os.listdir(chk_dir)
            if chk_file:
                chk_file = chk_file[0]
                return os.path.join(chk_dir, chk_file)
    except FileNotFoundError:
        return None

    return None


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def get_exp_num(EXP_DIR):
#     num=0
#     exp_name = os.path.join(EXP_DIR,str(num))
#     while is_folder_exist(exp_name):
#         num+=1
#         exp_name = os.path.join(EXP_DIR, str(num))
#     return num
