# In[1] Constants and params:
import json
import logging
import os
import random
import sys

import numpy as np
import torch
from argparse import ArgumentParser
from pathlib import Path

exp_number = None


def parse_args(channel_idx=0, exp_num=None):
    global exp_number
    exp_number = exp_num

    parser = ArgumentParser()

    # Data Configuration
    parser.add_argument('--channels', type=list,
                        default=['AGP', 'DNA', 'ER', 'Mito', 'RNA'],
                        help='the names of the different channels in the dataset')
    parser.add_argument('--families', type=list,
                        default=['Granularity', 'Intensity', 'Location', 'RadialDistribution', 'Texture'],
                        help="the features' families in the dataset")
    parser.add_argument('--label_field', type=str, default='Metadata_ASSAY_WELL_ROLE',
                        help='the field that contain the labels of the samples')
    parser.add_argument('--labels', type=list,
                        default=['mock', 'treated'],
                        help="the different labels of the samples in the dataset")
    parser.add_argument('--split_field', type=str, default='Image_Metadata_Well',
                        help='the field that split the train data')
    parser.add_argument('--sample_n', type=int, default=16,
                        help='Sample size for train in split')
    parser.add_argument('--cols_file', type=Path, default='/storage/users/g-and-n/plates/columns.txt',
                        help='json file containing a dictionary maps the different fields into channels')
    parser.add_argument('--index_fields', type=list,
                        default=['Plate', 'Metadata_ASSAY_WELL_ROLE', 'Metadata_broad_sample',
                                 'Image_Metadata_Well', 'ImageNumber', 'ObjectNumber'],
                        help='index fields of the samples in the dataset')
    parser.add_argument('--metadata_path', type=Path, default='/storage/users/g-and-n/plates/tabular_metadata.csv',
                        help='where to save/load from the metadata file')
    parser.add_argument('--plates_path', type=Path, default='/storage/users/g-and-n/plates/csvs/',
                        help="the folder that contains all the plates' csvs")

    # Data Loader Parameters
    parser.add_argument('--batch_size', type=int, default=1024 * 8,
                        help='the size of the batches to load')
    parser.add_argument('--num_data_workers', type=int, default=6,
                        help='number of workers for each data loader')

    # Training Configuration
    parser.add_argument('--train_labels', type=list,
                        default=['mock'],
                        help="on which labels of the samples in the dataset the model should train")
    parser.add_argument('--plates_split', type=list,
                        default=[[25732, 26564, 26572, 26575], [26569, 26574]],
                        help='plates split between train and test. left is train and right is test')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                        help='split ratio between train and validation. value in [0,1]')
    parser.add_argument('--input_channels', type=list,
                        default=['GENERAL', 'DNA', 'ER', 'Mito', 'RNA'],
                        help="channels that are the input for the model")
    parser.add_argument('--target_channels', type=list,
                        default=['AGP'],
                        help="channels that are the output of the model")
    parser.add_argument('--device', type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help='device for running code')
    parser.add_argument('--output_root_path', type=Path, default='/storage/users/g-and-n/tabular_models_results/',
                        help="where to save the experiment")

    # Training parameters
    parser.add_argument('-m', '--mode', type=str, default='train', choices=('train', 'predict'),
                        help='Whether to train a model or load a model and predict')
    parser.add_argument('--test_samples_per_plate', type=int, default=-1,
                        help='number of test samples for each plate. if None, all plates are taken')
    parser.add_argument('--seed', type=int, default=42,
                        help='global seed (for weight initialization, data sampling, etc.). ')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train')

    args, _ = parser.parse_known_args()

    args.cols_dict = json.load(open(args.cols_file, 'r'))

    in_channels = ['GENERAL'] + args.channels[:channel_idx] + args.channels[channel_idx + 1:]
    update_in_channels(in_channels, args)

    out_channels = [args.channels[channel_idx]]
    update_out_channels(out_channels, args)

    setup_logging(args)
    setup_determinism(args)

    return args


def update_in_channels(channels, args):
    args.input_channels = channels
    args.input_fields = sum([args.cols_dict[k] for k in args.input_channels], [])
    update_norm_pth(args)


def update_out_channels(channels, args):
    args.target_channels = channels
    args.target_fields = sum([args.cols_dict[k] for k in args.target_channels], [])
    update_norm_pth(args)

    args.exp_dir = os.path.join(args.output_root_path, str(exp_number), "channel " + '-'.join(args.target_channels))
    os.makedirs(args.exp_dir, exist_ok=True)
    args.checkpoint = get_checkpoint(args.exp_dir)


def update_norm_pth(args):
    n_pth = fr"/storage/users/g-and-n/plates/{'_'.join(args.input_channels)}-{'_'.join(args.target_channels)}.normsav"
    args.norm_params_path = n_pth


def setup_logging(args):
    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr)]
    logging.basicConfig(level=logging.INFO, format=head, style='{', handlers=handlers)
    # logging.info('Start with arguments {}'.format(args))


def setup_determinism(args):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def get_checkpoint(LOG_DIR):
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
