import logging
import os
import pickle
import sys
from time import time

import pandas as pd
import numpy as np
from itertools import combinations

from configuration.config import parse_args, update_in_channels, update_out_channels
from configuration.model_config import Model_Config
from data_layer.prepare_data import load_data
from model_layer.TabularAE import unify_test_function

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch


def main(model, args, kwargs={}):
    print_exp_description(model, args, kwargs)

    logging.info('Preparing data...')
    s = time()
    dataloaders = load_data(args)
    e = time()
    logging.info(f'Done loading data in {e - s} seconds')

    model = model.model_class(**model.params, batch_size=args.batch_size)
    if args.mode == 'predict' and args.checkpoint is not None:
        logging.info('loading model from file...')
        model = model.load_from_checkpoint(args.checkpoint)
        model.to(args.device)
        logging.info('loading model from file finished')

        logging.info('testing model...')
        model.eval()
        res = test_by_partition(model, dataloaders['test'], args.exp_dir)
        logging.info('testing model finished...')

        save_results(res, args, kwargs)

    elif args.mode == 'train':
        logging.info('training model...')
        model.to(args.device)
        logger = TensorBoardLogger(args.exp_dir,
                                   name='log_dir')
        trainer = pl.Trainer(max_epochs=args.epochs, progress_bar_refresh_rate=1, logger=logger,
                             gpus=int(torch.cuda.is_available()),
                             auto_scale_batch_size='binsearch', weights_summary='full')
        trainer.fit(model, dataloaders['train'], dataloaders['val_for_test'])
        logging.info('training model finished.')
    elif args.mode == 'evaluate':
        logging.info('loading model from file...')
        model = model.load_from_checkpoint(args.checkpoint)
        model.to(args.device)
        logging.info('loading model from file finished')

        # Create Trainer to evaluate the model
        logger = TensorBoardLogger(args.exp_dir,
                                   name='log_dir')
        trainer = pl.Trainer(max_epochs=args.epochs, progress_bar_refresh_rate=1, logger=logger,
                             gpus=int(torch.cuda.is_available()),
                             auto_scale_batch_size='binsearch', weights_summary='full')

        res = []

        # Evaluate the train and validation datasets, will get the metrics from validation_step in the model
        train_val_metrics = trainer.validate(model, [dataloaders['train'], dataloaders['val_for_test']])
        for i, (dataset, metrics) in enumerate(zip(['train', 'val'], train_val_metrics)):
            mse = metrics[f'val_loss_epoch/dataloader_idx_{i}']
            pcc = metrics[f'val_pcc_epoch/dataloader_idx_{i}']
            res.append([dataset, None, None, mse, pcc])

        # Collecting the test datasets
        test_loaders = []
        test_loaders_idx = []
        for plate, plate_data in dataloaders['test'].items():
            for subset, dataloader in plate_data.items():
                test_loaders_idx.append((plate, subset))
                test_loaders.append(dataloader)

        # Evaluate the test datasets
        test_metrics = trainer.validate(model, test_loaders)
        for i, ((plate, subset), metrics) in enumerate(zip(test_loaders_idx, test_metrics)):
            mse = metrics[f'val_loss_epoch/dataloader_idx_{i}']
            pcc = metrics[f'val_pcc_epoch/dataloader_idx_{i}']
            res.append(['test', plate, subset, mse, pcc])

        # Export results
        res_df = pd.DataFrame(res, columns=['Dataset', 'Plate', 'Subset', 'MSE', 'PCC'])
        save_results(res_df, args)
        logging.info('Finished metrics evaluation')
        print(res_df)


def print_exp_description(Model, args, kwargs):
    description = 'Training Model ' + Model.name + ' with target ' + '-'.join(args.target_channels)
    for arg in kwargs:
        description += ', ' + arg + ': ' + str(kwargs[arg])
    print(description)

    print('Arguments:')
    col = 3
    i = 0
    for k, v in args.__dict__.items():
        v = str(v)
        v = v if len(v) < 100 else v[:50] + '...' + v[-50:]
        print(f'\t{k}: {v}', end='')
        i = (i + 1) % col
        if not i:
            print()
    print()


def test_by_partition(model, test_dataloaders, exp_dir):
    result_path = os.path.join(exp_dir, 'results')
    os.makedirs(result_path, exist_ok=True)

    results = []
    for plate, plate_data in test_dataloaders.items():
        res_plate_path = os.path.join(result_path, f'{plate}.csv')
        if not os.path.exists(res_plate_path):
            plate_results = []
            for _, dataloader in test_dataloaders[plate].items():
                plate_res = test(model, dataloader)
                plate_results.append(plate_res)

            plate_res = pd.concat(plate_results)
            plate_res.to_csv(res_plate_path, index=False)
            del plate_res
            del plate_results
        # else:
        #     plate_res = pd.read_csv(res_plate_path)

        # results.append(plate_res)

    # res = pd.concat(results)
    return pd.DataFrame()


def test(model, data_loader):
    pred, mse, pcc = zip(*[unify_test_function(model, batch, mse_reduction='none') for batch in data_loader])
    mses = [i.cpu().numpy() for i in mse]
    mses = np.concatenate(mses)
    index = data_loader.dataset.get_index()
    rec = np.concatenate([index.to_numpy(), mses], axis=1)

    results = pd.DataFrame(rec, columns=data_loader.dataset.index_fields + data_loader.dataset.target_fields)
    return results


def save_results(res, args, kwargs={}):
    for arg in kwargs:
        res[arg] = kwargs[arg]

    res_dir = os.path.join(args.exp_dir, 'results.csv')
    if os.path.isfile(res_dir):
        try:
            prev_res = pd.read_csv(res_dir)
            res = pd.concat([prev_res, res])
        except:
            print('Could not load old result.. overwriting')

    # save_to_pickle(res, os.path.join(args.exp_dir, 'results.pkl'))
    save_to_pickle(args, os.path.join(args.exp_dir, 'args.pkl'))
    res.to_csv(res_dir, index=False)


def save_to_pickle(obj, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


if __name__ == '__main__':
    inp = int(sys.argv[1])
    # model_id = inp // 100
    # channel_id = (inp // 100) - 1
    lsd = 8  # 2 ** (inp % 10)
    # plate_id = (inp % 100) - 1
    channel_id = 0
    # split_c = (inp // 10) - 1

    channels_inout = [
        (['AGP'], ['DNA', 'Mito']),
        (['AGP'], ['DNA', 'RNA']),
        (['AGP'], ['ER', 'Mito']),
        (['AGP'], ['ER', 'RNA']),
        (['DNA'], ['AGP', 'Mito']),
        (['DNA'], ['AGP', 'RNA']),
        (['ER'], ['AGP', 'Mito']),
        (['ER'], ['AGP', 'RNA']),
        (['Mito'], ['AGP', 'DNA']),
        (['Mito'], ['AGP', 'ER']),
        (['RNA'], ['AGP', 'DNA']),
        (['RNA'], ['AGP', 'ER']),
    ]

    exp_num = 42100 + inp  # if None, new experiment directory is created with the next available number
    DEBUG = False

    exps = [(lr, batch_size)
            for batch_size in [512, 1024, 2048, 4096, 8196]
            for lr in [1.5e-4, 1.0e-4, 1.5e-3, 1.0e-3, 1.5e-2, 1.0e-2]
            ]
    exp_values = exps[27 - 1]

    lr, batch_size = exp_values
    exp_dict = {'lr': lr,
                'epochs': 20, 'latent_space_dim': lsd}

    channels_to_predict = [channel_id]

    model = Model_Config.TAE

    for target_channel in channels_to_predict:
        # torch.cuda.empty_cache()
        args = parse_args(channel_idx=target_channel, exp_num=exp_num)
        args.batch_size = batch_size
        args.epochs = exp_dict['epochs']

        out_chans, in_chans = channels_inout[inp-1]
        # in_chans = [args.channels[target_channel]]
        # in_channels = ['GENERAL'] + list(list(combinations(in_chans, 3))[split_c])
        update_in_channels(['GENERAL'] + in_chans, args)
        update_out_channels(out_chans, args)

        exp_dict['input_size'] = len(args.input_fields)
        exp_dict['target_size'] = len(args.target_fields)
        model.update_custom_params(exp_dict)

        # args.mode = 'train'
        args.mode = 'predict'
        args.mode = 'evaluate'

        plates35 = [24792, 25912, 24509, 24633, 25987, 25680, 25422,
                    24517, 25664, 25575, 26674, 25945, 24687, 24752,
                    24311, 26622, 26641, 24594, 25676, 24774, 26562,
                    25997, 26640, 24562, 25938, 25708, 24321, 24735,
                    26786, 25571, 26666, 24294, 24640, 25985, 24661]
        # 24278

        plates_lte3 = [[26598, 24785, 24792, 24525, 26166, 24518, 25923, 24740, 24797, 24507,
                        25374, 26542, 24796, 25725, 26531, 24789, 26575, 25742, 26626, 26544,
                        25372, 25553, 24750, 25686, 26579, 25675, 25683, 24623, 24618, 24508,
                        25674, 25676, 24619, 26562, 26592, 26600, 24666, 24683, 25564, 25708,
                        26521, 26138, 25707, 25376, 26572, 24667, 24795, 25503, 25726, 26578,
                        25739, 26596, 26576, 26574, 26569, 26564, 25738, 26580, 26577, 25915,
                        26271, 26563, 25688, 25741, 26595, 25918, 24523, 25732, 26588, 24509,
                        24751],
                       [24596, 26679, 24514, 26159, 26672, 25848, 25410, 25967, 24657, 26642, 25962, 26678, 24293,
                        24296, 25931, 24586, 24773, 24633, 26205, 24308, 25642, 25391, 26668, 26663, 25692, 24754,
                        24584, 24609, 24307, 26644, 24793, 24300, 24305, 24617, 26008, 25991, 24278, 25925, 25567,
                        25588, 24277, 25592, 24651, 24752, 25584, 25639, 26688, 25856, 25857, 25664, 25859, 25681,
                        26607, 26071, 25576, 24655, 24310, 25418, 25414, 26124, 25937, 26666, 25380, 25571, 24648,
                        26702, 24634, 24753, 26795, 24736, 26007, 24605, 25565, 26685, 25740, 25593, 24732, 25966,
                        25912, 24774, 26006, 24312, 24303, 26675, 26670, 25566, 25665, 24731, 26608, 25990, 24640,
                        26641, 24566, 25986, 24684, 26092, 24280, 26216, 26794, 25382, 25403, 26207, 25847, 25943,
                        26174, 25849, 25430, 25406, 25598, 25594, 24583, 25904, 24653, 25605, 25643, 25994, 26203,
                        26060, 26110, 24624, 25428, 24631, 26625, 24311, 26643, 26133, 26771, 24306, 24564, 24302,
                        24563, 24663, 26684, 26622, 24733, 26744, 24734, 26058, 24516, 26753, 25724, 25488, 26739,
                        24517, 25997, 24772, 24301, 26695, 25852, 26673, 25853, 24739, 26640, 26009, 24357, 24611,
                        24602, 25914, 25892, 24297, 25890, 24515, 25580, 24625, 24685, 25704, 25680, 25983, 25591,
                        25911, 25854, 26118, 25694, 25992, 24638, 25938, 24647, 24636, 25700, 26681, 24645, 24641,
                        26181, 25903, 26703, 24304, 26611, 24637, 24646, 25432, 24593, 24590, 26705, 26140, 24595,
                        25955, 25968, 24644, 24320, 25939, 24592, 26545, 24642, 26683, 24755, 25695, 25578, 25690,
                        25641, 25855, 25677, 24604, 24594, 25568, 26662, 24735, 25485, 26748, 24775, 26204, 25435,
                        25929, 25587, 25993, 25438, 26765, 25913, 24661, 25426, 26677, 25985, 25599, 25573, 25862,
                        24309, 25490, 24313, 24565, 25965, 26081, 25583, 26135, 24726, 25570, 25908, 24321, 24512,
                        25424, 25422, 26002, 24295, 25944, 25579, 25987, 25667, 26745, 25387, 26767, 25891, 25436,
                        24294, 26669, 25678, 24664, 26126, 26107, 24588, 26786, 24352, 25378, 24783, 26671, 26612,
                        26730, 24279, 25392, 26202, 26601, 26768, 26232, 26623, 25569, 25989, 25581, 26682, 26664,
                        24319, 26674, 25949, 24759, 26752, 24756, 26724, 25420, 24656, 25689, 26224, 24643, 25984,
                        26115, 24562, 26061, 24585, 26128, 25572, 25663, 24688, 26239, 25988, 25575, 25679, 26785,
                        24639, 24687, 24560, 24652, 25858, 25945, 25935, 25638, 25585, 25434, 24758, 25416, 25492,
                        24654, 26680, 25590, 25909, 24591, 26247, 25885, 26772, 24635, 25408]]

        plates100 = [
            24596, 26679, 24514, 26159, 26672, 25848, 24657, 26642, 25962, 26678, 26626,
            25681, 26607, 26071, 25576, 25418, 25380, 25571, 24648,
            25726, 26007, 26596, 24605, 25565, 26685, 25593, 24732,
            25966, 25912, 24684, 25904, 26580, 24653, 25605,
            25643, 24740, 26625, 24311, 26643, 26133, 26771, 24306,
            24647, 25700, 26681, 24645, 26611, 24637, 24646, 25432, 24593,
            24590, 26705, 26140, 24595, 24604, 24594, 25568, 26662, 24735,
            26748, 24775, 24623, 25993,
            24661, 25426, 26677, 25985, 25915, 24313, 24565, 25965, 26081,
            25374, 26745, 26531, 24789, 25387, 24294,
            26682, 26664, 24319, 26674, 24759, 24751, 26752, 24656, 25689, 26224, 24643,
            25984, 24562, 26061, 25575, 26785,
            24639, 26592, 24666, 24652, 25858
        ]
        all_plates = [
            24596, 26679, 24514, 26159, 26672, 25848, 25410, 25967, 24657, 26642, 25962, 26678, 24293, 24296, 26626,
            25931, 24586, 24773, 24633, 26205, 24308, 25642, 25391, 25708, 26668, 26663, 25692, 24754, 24584, 24609,
            24307, 25503, 26644, 24793, 24300, 24305, 24617, 26008, 25738, 25991, 24278, 25925, 25567, 26271, 25588,
            24277, 26588, 24509, 25592, 24651, 24752, 25584, 26598, 25639, 24785, 26688, 25856, 25857, 25664, 25859,
            25681, 26607, 26071, 25576, 24655, 24310, 25418, 26579, 25414, 26124, 25937, 26666, 25380, 25571, 24648,
            26702, 24634, 24753, 26795, 24736, 25726, 26007, 25739, 26596, 24605, 25565, 26685, 25740, 25593, 24732,
            25966, 25912, 24774, 26006, 24312, 24303, 26675, 26670, 25566, 26542, 25665, 24731, 26608, 25990, 24640,
            26641, 26544, 24566, 25683, 25986, 24684, 26092, 24508, 24280, 26216, 26794, 25382, 25403, 24683, 26207,
            25847, 25943, 26174, 25849, 26572, 24795, 25430, 25406, 25598, 25594, 24583, 25904, 26580, 24653, 25605,
            25643, 25994, 26203, 26060, 26110, 24624, 25428, 24631, 24740, 26625, 24311, 26643, 26133, 26771, 24306,
            24796, 24564, 24302, 24563, 24663, 26684, 26622, 24733, 25553, 26744, 24734, 26058, 25674, 26562, 25676,
            24516, 26753, 25724, 25488, 26739, 24517, 25997, 24772, 24301, 26695, 25852, 26673, 25853, 24739, 26640,
            26564, 26009, 24357, 24611, 25688, 26595, 24602, 25914, 25892, 24297, 25890, 24515, 25580, 24625, 24685,
            25923, 24797, 24507, 25704, 25680, 25983, 25591, 25911, 25854, 26575, 26118, 25694, 25992, 24638, 25938,
            24647, 24636, 25700, 26681, 24645, 24641, 26181, 25903, 26703, 24304, 26611, 24637, 24646, 25432, 24593,
            24590, 26705, 26140, 24595, 25918, 25955, 25968, 24644, 24320, 25939, 24592, 26545, 26166, 24642, 24518,
            26683, 24755, 25695, 25578, 25690, 25641, 25855, 25677, 24604, 24594, 25568, 26662, 24735, 25372, 25485,
            26748, 24775, 24750, 26204, 24623, 25435, 25929, 25587, 25993, 25564, 25438, 26521, 26765, 25913, 25376,
            24661, 25426, 26677, 25985, 25599, 25573, 25862, 24309, 25490, 25915, 24313, 24565, 25965, 26081, 25732,
            24523, 25583, 26135, 24726, 25570, 25908, 24321, 24512, 25424, 25422, 26002, 24792, 24525, 24295, 25944,
            25579, 25987, 25374, 25667, 25725, 26745, 26531, 24789, 25387, 26767, 25891, 25686, 25436, 24294, 26669,
            25678, 25675, 24664, 24619, 26126, 26107, 26600, 24588, 26786, 24352, 25378, 25707, 24783, 26671, 26612,
            26730, 24279, 25392, 24667, 26202, 26601, 26768, 26232, 26623, 26576, 25569, 26569, 26574, 25989, 25581,
            26682, 26664, 24319, 26674, 25949, 24759, 24751, 26752, 24756, 26724, 25420, 24656, 25689, 26224, 24643,
            25984, 26115, 24562, 26061, 24585, 26128, 25572, 25663, 24688, 25742, 26239, 25988, 25575, 25679, 26785,
            24639, 24687, 24618, 24560, 26592, 24666, 24652, 25858, 26138, 25945, 25935, 25638, 25585, 25434, 24758,
            26578, 25416, 25492, 24654, 26680, 26577, 25590, 25909, 24591, 25741, 26247, 25885, 26772, 24635, 25408,
            26563]

        all_split = [all_plates, all_plates.copy()]

        args.plates_split = [plates100, plates100.copy()]
        # args.plates_split = [
        #     [p for p in plates if p != plates[plate_id]],
        #     [plates[plate_id]]
        # ]
        args.split_ratio = 0.8

        args.test_samples_per_plate = None

        if DEBUG:
            args.test_samples_per_plate = 10
            args.epochs = 5

        main(model, args)
