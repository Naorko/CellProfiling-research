import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from data_layer.create_tabular_metadata import create_tabular_metadata
from data_layer.tabular_dataset_with_processed import TabularDataset

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


def load_data(args):
    """
    The main function to load all the dataset into dataloaders
    :param args: Namespace object of the configurations
    :return: list of dataloaders
    """

    # Load metadata pd.Dataframe
    all_plates = args.plates_split[0]
    if args.metadata_path is None:
        mt_df = create_tabular_metadata(args.plates_path, all_plates, args.label_field, args.train_labels,
                                        args.split_field, args.sample_n)
        mt_df.to_csv(args.metadata_path, index=False)
    else:
        mt_df = pd.read_csv(args.metadata_path, dtype={'Plate': int, 'Count': int})

        # TODO - fix reproduce metadata table and remove following line
        mt_df.loc[mt_df['Metadata_ASSAY_WELL_ROLE'].isna(), 'Metadata_ASSAY_WELL_ROLE'] = 'mock'

        mt_df[args.split_field] = mt_df[args.split_field].apply(eval)
        plates = [p for p in all_plates if p not in mt_df['Plate'].unique()]
        if plates:
            add_df = create_tabular_metadata(args.plates_path, plates, args.label_field, args.train_labels,
                                             args.split_field, args.sample_n)
            mt_df = pd.concat([mt_df, add_df], ignore_index=True)
            mt_df.to_csv(args.metadata_path, index=False)

    # Split the plates into train\val\test partitions of indexes
    partitions = split_by_plates(mt_df, args)

    # Transform the indexes to sub-dataframes of the metadata
    partitions = partitions_idx_to_dfs(mt_df, partitions)

    # Create dataset object for each partition
    datasets = create_datasets(args.plates_split, partitions, args.plates_path,
                               args.input_fields, args.target_fields, args.index_fields,
                               args.device, args.norm_params_path)
    print_data_statistics(datasets)
    # Create dataloader object for each dataset object
    data_loaders = create_data_loaders(datasets, partitions, args.batch_size, args.num_data_workers)

    return data_loaders


def split_by_plates(df, args) -> dict:
    """
    Split the plates into train\val\test partitions of indexes
    :param df: metadata pd.Dataframe
    :param args: Namespace configurations object
    :return: partitions including the line indexes of the original metadata dataframe
    """
    train_plates, test_plates = args.plates_split
    # train_plates, val_plates = train_test_split(train_plates, train_size=args.split_ratio, shuffle=True)
    val_plates = train_plates.copy()

    logging.info(f'Train Plates: {" ".join(str(t) for t in train_plates)}')
    logging.info(f'Validation Plates: {" ".join(str(t) for t in val_plates)}')
    logging.info(f'Test Plates: {" ".join(str(t) for t in test_plates)}' if test_plates else 'There are no test plates')

    partitions = {
        'train': list(df[(df['Plate'].isin(train_plates)) & (df[args.label_field].isin(args.train_labels)) & (
                df['Mode'] == 'train')].index),
        'val': list(df[(df['Plate'].isin(val_plates)) & (df[args.label_field].isin(args.train_labels)) & (
                df['Mode'] == 'val')].index),
        'test': {}
    }

    if test_plates is None:
        test_plates = train_plates

    # divide test data into plates (mock, irradiated and active from test plates)
    for plate in test_plates:
        partitions['test'][str(plate)] = {}
        for lbl in args.labels:
            partitions['test'][str(plate)][lbl] = list(
                df[(df['Plate'] == plate) & (df[args.label_field] == lbl) & (df['Mode'] == 'predict')].index)[
                                                  :args.test_samples_per_plate]

    return partitions


def partitions_idx_to_dfs(mt_df, partitions):
    """
    Transform the indexes to sub-dataframes of the metadata
    :param mt_df: the original metadata dataframe
    :param partitions: the partitions including the line indexes
    :return: partitions of sub-dataframes of the original metadata dataframe
    """
    df_partitions = {
        'train': mt_df.iloc[partitions['train']].copy(),
        'val': mt_df.iloc[partitions['val']].copy(),
        'test': {}
    }

    for plate in list(partitions['test'].keys()):
        df_partitions['test'][plate] = {}
        for key in partitions['test'][plate].keys():
            test_idxs = partitions['test'][plate][key]
            if test_idxs:
                df_partitions['test'][plate][key] = mt_df.iloc[test_idxs].copy()

    return df_partitions


def get_tensor(inp):
    """
    Tensor transformer
    :param inp: input array
    :return: Torch tensor
    """
    return torch.from_numpy(inp)


def get_normalizer(mean, std):
    """
    Normalizer transformer
    :param mean: The mean of the inputs
    :param std: The standard deviation of the input
    :return: The normalized input
    """
    def normalizer(inp):
        return inp.sub_(mean).div_(std)

    return normalizer


def create_datasets(plates_split, partitions, data_dir,
                    input_fields, target_fields, index_fields,
                    device, norm_params_path):
    """
    Create dataset object for each partition
    :param plates_split: tuple (train plates, test plates)
    :param partitions: partitions of the metadata dataframe
    :param data_dir: path to the plates' folder
    :param input_fields: subset of the fields used for input
    :param target_fields: subset of the fields used for output
    :param index_fields: subset of the fields used for index
    :param device: device to use (cpu/gpu)
    :param norm_params_path: path to save/load the train statistics
    :return: partitions of datasets
    """
    train_plates, test_plates = plates_split
    mean, std = get_data_stats(partitions['train'], train_plates, data_dir, device,
                               input_fields, target_fields, norm_params_path, index_fields)
    mean, std = [torch.tensor(lst, dtype=torch.float32) for lst in [mean, std]]

    tabular_transforms = transforms.Compose([
        get_tensor,
        get_normalizer(mean, std)
    ])

    datasets = {
        'train': TabularDataset(partitions['train'], root_dir=data_dir, transform=tabular_transforms,
                                input_fields=input_fields, target_fields=target_fields, index_fields=index_fields,
                                shuffle=True),
        'val': TabularDataset(partitions['val'], root_dir=data_dir, transform=tabular_transforms,
                              input_fields=input_fields, target_fields=target_fields, index_fields=index_fields,
                              shuffle=False),
        'val_for_test': TabularDataset(partitions['val'], root_dir=data_dir, transform=tabular_transforms,
                                       input_fields=input_fields, target_fields=target_fields,
                                       index_fields=index_fields,
                                       is_test=True, shuffle=False),
        'test': {}
    }

    for plate in list(partitions['test'].keys()):
        datasets['test'][plate] = {}
        for key in partitions['test'][plate].keys():
            datasets['test'][plate][key] = \
                TabularDataset(partitions['test'][plate][key], root_dir=data_dir, transform=tabular_transforms,
                               input_fields=input_fields, target_fields=target_fields, index_fields=index_fields,
                               is_test=True, shuffle=False)

    return datasets


def print_data_statistics(datasets):
    """
    Print statistics over the datasets partitions
    :param datasets:
    :return: None
    """
    print(f'train set contains {len(datasets["train"])} cells')
    print(f'val set contains {len(datasets["val"])} cells')

    test_len = {}
    for plate in list(datasets['test'].keys()):
        for key in datasets['test'][plate].keys():
            cur_cnt = test_len.get(key, 0)
            cur_cnt += len(datasets['test'][plate][key])
            test_len[key] = cur_cnt

    for key, cnt in test_len.items():
        print(f' test set of {key} contains {cnt} cells')


def get_data_stats(train_mt_df, train_plates, data_dir, device,
                   input_fields, target_fields, norm_params_path, index_fields):
    """

    :param train_mt_df: train metadata dataframe
    :param train_plates: list of plates for train
    :param data_dir: path to the plates' folder
    :param device: device to use (cpu/gpu)
    :param input_fields: subset of the fields used for input
    :param target_fields: subset of the fields used for output
    :param norm_params_path: path to save/load the train statistics
    :param index_fields: subset of the fields used for index
    :return: mean and standard deviation of the train data
    """
    if os.path.exists(norm_params_path):
        mean, std = joblib.load(norm_params_path)
    else:
        logging.info('calculating mean and std...')
        mean, std = calc_mean_and_std(train_mt_df, data_dir, len(train_plates), device, input_fields, target_fields,
                                      index_fields)
        joblib.dump((mean, std), norm_params_path)

    return mean, std


def calc_mean_and_std(mt_df, data_dir, num_batches, device, input_fields, target_fields, index_fields):
    """
    Calculate the mean and standard deviation of the train data
    :param mt_df: metadata dataframe
    :param data_dir: path to the plates' folder
    :param num_batches: nuber of batches to calculate by
    :param device: device to use (cpu/gpu)
    :param input_fields: subset of the fields used for input
    :param target_fields: subset of the fields used for output
    :param index_fields: subset of the fields used for index
    :return: mean and standard deviation of the train data
    """
    train_data = TabularDataset(mt_df, root_dir=data_dir,
                                input_fields=input_fields, target_fields=target_fields,
                                for_data_statistics_calc=True, index_fields=index_fields)
    batch_size = int(len(train_data) / num_batches)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    num_channels = len(input_fields) + len(target_fields)

    mean = torch.zeros(num_channels).to(device)
    std = torch.zeros(num_channels).to(device)
    max_p = 0
    min_p = 65535

    for ind, samples in train_loader:
        samples = samples.to(device)
        batch_mean, batch_std = torch.std_mean(samples.float(), dim=(0,))
        max_p = max(torch.max(samples.float()), max_p)
        min_p = min(torch.min(samples.float()), min_p)

        mean += batch_mean
        std += batch_std

    mean /= num_batches
    std /= num_batches
    print('mean of train data is ' + str(mean.tolist()))
    print('std of train data is ' + str(std.tolist()))
    print('maximum of train data is ' + str(max_p.tolist()))
    print('minimum of train data is ' + str(min_p.tolist()))

    return mean.tolist(), std.tolist()


def create_data_loaders(datasets, partitions, batch_size, num_workers=32) -> dict:
    """
    Create dataloader object for each dataset object
    :param datasets: partitions of datasets objects
    :param partitions: partitions of metadata dataframes
    :param batch_size:
    :param num_workers: number of workers for each dataloader
    :return: partitions of dataloader
    """
    data_loaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size,
                            shuffle=False, num_workers=num_workers),
        'val': DataLoader(datasets['val'], batch_size=batch_size,
                          shuffle=False, num_workers=num_workers),
        'val_for_test': DataLoader(datasets['val_for_test'], batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers),
        'test': {}
    }

    # from time import time
    # s=time()
    # batch = next(iter(data_loaders['train']))
    # print(f'Took {time()-s} seconds')
    # exit(42)

    for plate in list(partitions['test'].keys()):
        data_loaders['test'][plate] = {}
        for key in partitions['test'][plate].keys():
            data_loaders['test'][plate][key] = \
                DataLoader(datasets['test'][plate][key], batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)

    return data_loaders
