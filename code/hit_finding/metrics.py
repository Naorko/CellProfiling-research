from itertools import cycle
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from glob import glob

import os
import sys

sys.path.append(os.path.abspath('..'))
from hit_finding.constants import *
from learning_tabular.constants import CHANNELS, LABEL_FIELD
from learning_tabular.preprocessing import load_plate_csv, list_columns


def load_pure_zscores(plate_csv, raw=False, by_well=True, inter_channel=True, index_fields=None, well_index=None,
                      dest=None):
    if dest is None:
        if raw:
            if inter_channel:
                dest = 'raw'
            else:
                dest = 'raw1to1'
        else:
            dest = 'err'

        dest = f'{pure_fld}/{dest}/{os.path.basename(plate_csv)}'

    if os.path.exists(dest):
        index_size = 4 if by_well else 6
        return pd.read_csv(dest, index_col=list(range(index_size)))

    df = load_plate_csv(plate_csv, index_fields=index_fields)

    if by_well:
        if well_index is None:
            well_index = ['Plate', LABEL_FIELD, 'Metadata_broad_sample', 'Image_Metadata_Well']

        df = df.groupby(by=well_index).apply(lambda g: g.mean())

    df_mock = df[df.index.isin(['mock'], 1)]
    if not df_mock.shape[0]:
        print(f'no mock wells in {os.path.basename(plate_csv)}')
        return None

    scaler = StandardScaler()
    scaler.fit(df_mock)
    del df_mock

    df_zscores = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
    del df

    df_zscores.to_csv(dest)

    return df_zscores


def extract_pure(plate_csv, index_fields=None, well_index=None):
    print('.', end='')

    err = load_pure_zscores(f'{err_fld}/{plate_csv}', raw=False,
                            index_fields=index_fields, well_index=well_index)
    del err
    # raw = load_pure_zscores(f'{raw_fld}/{plate_csv}', raw=True)
    # del raw
    raw1to1 = load_pure_zscores(f'{raw1to1_fld}/{plate_csv}', raw=True, inter_channel=False,
                                index_fields=index_fields, well_index=well_index)
    del raw1to1


def extract_z_score(plate_csv, by_well=True, by_channel=True, abs_zscore=True, well_type='treated', raw=False,
                    inter_channel=True):
    df = load_pure_zscores(plate_csv, raw, inter_channel)

    if well_type in ['treated', 'mock']:
        df_selected = df[df.index.isin([well_type], 1)]
        del df
    else:
        df_selected = df

    if abs_zscore:
        df_selected = df_selected.abs()

    if by_channel:
        _, _, channels = list_columns(df_selected)
        for channel, cols in channels.items():
            df_selected[channel] = df_selected[cols].mean(axis=1)

        channels_cols = [col for ch_cols in channels.values() for col in ch_cols]
        df_selected["ALL"] = df_selected[channels_cols].mean(axis=1)

        data = df_selected[CHANNELS + ["ALL"]]
        del df_selected
    else:
        data = df_selected

    if by_well:
        gb = data.groupby(by=['Plate', 'Metadata_broad_sample', 'Image_Metadata_Well'])
        del data

        by_trt = gb.apply(lambda g: g.mean())
        return by_trt

    return data


def extract_score(plate_csv, by_well=True, by_channel=True, abs_zscore=True, well_type='treated', raw=False, thresh=4,
                  inter_channel=True):
    df = load_pure_zscores(plate_csv, raw, inter_channel)

    if well_type in ['treated', 'mock']:
        df_selected = df[df.index.isin([well_type], 1)]
        del df
    else:
        df_selected = df

    if abs_zscore:
        df_selected = df_selected.abs()

    df_selected = df_selected.apply(lambda x: x.apply(lambda y: 0 if y < thresh else 1))

    if by_channel:
        _, _, channels = list_columns(df_selected)
        for channel, cols in channels.items():
            df_selected[channel] = df_selected[cols].sum(axis=1) / len(cols)

        channels_cols = [col for ch_cols in channels.values() for col in ch_cols]
        df_selected["ALL"] = df_selected[channels_cols].sum(axis=1) / len(channels_cols)

        data = df_selected[CHANNELS + ["ALL"]]
        del df_selected
    else:
        data = df_selected

    if by_well:
        gb = data.groupby(by=['Plate', 'Metadata_broad_sample', 'Image_Metadata_Well'])
        del data

        by_trt = gb.apply(lambda g: g.mean())
        return by_trt

    return data


def extract_dist_score(plate_csv, well_type='treated', **kwargs):
    df = load_plate_csv(plate_csv)
    df = df.groupby(by=['Plate', LABEL_FIELD, 'Metadata_broad_sample', 'Image_Metadata_Well']).apply(
        lambda g: g.mean())

    def calculate_distance_from(v):
        return lambda x: np.linalg.norm(x - v)

    _, _, channels = list_columns(df)
    all_cols = [col for ch_cols in channels.values() for col in ch_cols]
    channels['ALL'] = all_cols

    scores = []
    for channel, cols in channels.items():
        df_mck = df[df.index.isin(['mock'], 1)][cols]
        mck_profile = df_mck.median()
        df_trt = df[df.index.isin([well_type], 1)][cols]

        dist_func = calculate_distance_from(mck_profile)
        trt_dist = df_trt.apply(dist_func, axis=1)
        del df_trt

        trt_dist.name = channel

        scores.append(trt_dist)

    del df

    scores_df = pd.concat(scores, axis=1)
    return scores_df


def extract_dist_score_norm_before(plate_csv, well_type='treated', **kwargs):
    df = load_pure_zscores(plate_csv, kwargs['raw'], kwargs['inter_channel'])

    def calculate_distance_from(v):
        return lambda x: np.linalg.norm(x - v)

    _, _, channels = list_columns(df)
    all_cols = [col for ch_cols in channels.values() for col in ch_cols]
    channels['ALL'] = all_cols

    scores = []
    for channel, cols in channels.items():
        df_mck = df[df.index.isin(['mock'], 1)][cols]
        mck_profile = df_mck.median()
        df_trt = df[df.index.isin([well_type], 1)][cols]

        dist_func = calculate_distance_from(mck_profile)
        trt_dist = df_trt.apply(dist_func, axis=1)
        del df_trt

        trt_dist.name = channel

        scores.append(trt_dist)

    del df

    scores_df = pd.concat(scores, axis=1)
    return scores_df


def extract_dist_score_norm_after(plate_csv, well_type='treated', **kwargs):
    df = load_plate_csv(plate_csv)
    df = df.groupby(by=['Plate', LABEL_FIELD, 'Metadata_broad_sample', 'Image_Metadata_Well']).apply(
        lambda g: g.mean())

    def calculate_distance_from(v):
        return lambda x: np.linalg.norm(x - v)

    _, _, channels = list_columns(df)
    all_cols = [col for ch_cols in channels.values() for col in ch_cols]
    channels['ALL'] = all_cols

    scores = []
    for channel, cols in channels.items():
        df_mck = df[df.index.isin(['mock'], 1)][cols]
        mck_profile = df_mck.median()
        df_trt = df[df.index.isin([well_type], 1)][cols]

        dist_func = calculate_distance_from(mck_profile)
        mck_dist = df_trt.apply(dist_func, axis=1)
        del df_mck
        trt_dist = df_trt.apply(dist_func, axis=1)
        del df_trt

        scaler = StandardScaler()
        scaler.fit(mck_dist.to_numpy().reshape(-1, 1))
        del mck_dist

        cur_scores = pd.Series(scaler.transform(trt_dist.to_numpy().reshape(-1, 1)).reshape(-1),
                               index=trt_dist.index,
                               name=channel)
        del trt_dist

        scores.append(cur_scores)

    del df

    scores_df = pd.concat(scores, axis=1)
    return scores_df


def extract_raw_and_err(score_func, plate_csv, by_well=True, well_type='treated', threshold=None):
    print('.', sep='', end='')

    params = {'by_well': by_well,
              'well_type': well_type,
              }
    if threshold is not None:
        params['thresh'] = threshold

    err = score_func(f'{err_fld}/{plate_csv}', abs_zscore=False, raw=False, inter_channel=True, **params)
    raw = score_func(f'{raw_fld}/{plate_csv}', abs_zscore=True, raw=True, inter_channel=True, **params)

    res = err.join(raw, how='inner', lsuffix='_map', rsuffix='_raw')
    del err
    del raw

    raw1to1 = score_func(f'{raw1to1_fld}/{plate_csv}', abs_zscore=False, raw=True, inter_channel=False, **params)
    res = res.join(raw1to1.add_suffix('_raw1to1'), how='inner')
    del raw1to1

    return res


def extract_scores_from_all(score_func, by_well=True, well_type='treated', threshold=None):
    p = Pool(3)

    plates = [f[1] for f in files]
    score_results = p.starmap(extract_raw_and_err,
                              zip(cycle([score_func]), plates, cycle([by_well]), cycle([well_type]),
                                  cycle([threshold])))
    p.close()
    p.join()

    scores = {plate_number: score_results[i] for i, plate_number in enumerate([f[0] for f in files])}
    return scores


if __name__ == '__main__':
    print('metrics main')
    print("Usage: metrics.py -p [experiment_path] [plate_index]")
    if sys.argv[1] == '-p':  # Means to run pure zscores run
        exp_fld = sys.argv[2]
        plates = glob(os.path.join(exp_fld, '*', 'results', '*'))
        try:
            plate_idx = int(sys.argv[3])
            plate = plates[plate_idx]
            pure_fld = os.path.join(exp_fld, 'zscores')
            os.makedirs(pure_fld, exist_ok=True)
            dest = os.path.join(pure_fld, os.path.basename(plate))
            print(f'Extract pure z-scores for plate {plate}')
            load_pure_zscores(plate, by_well=True,
                              index_fields=None,
                              well_index=None,
                              dest=dest)
            print('Done!')
        except:
            print(f'Error while reading plate {sys.argv[3]}')

    # For Visual Results
    # idx_fld = ['Plate', 'Well_Role', 'Broad_Sample', 'Well', 'Site', 'ImageNumber']
    # wll_idx = ['Plate', 'Well_Role', 'Broad_Sample', 'Well']

