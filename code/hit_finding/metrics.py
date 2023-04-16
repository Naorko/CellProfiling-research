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

"""
This file holds the calculations for various metrics to measure the treatments 
"""


def load_pure_zscores(plate_csv, raw=False, by_well=True, inter_channel=True, index_fields=None, well_index=None,
                      dest=None):
    """
    Load or create and save a normalized version of the input plate
    :param plate_csv: path to a plate's csv
    :param raw: whether it's raw features file or prediction (to calculate dest)
    :param by_well: whether to aggregate by well
    :param inter_channel: if raw=True, whether its raw features or predictions of 1to1 (to calculate dest)
    :param index_fields: the fields in plate_csv to be used as index
    :param well_index: if by_well=True, list of fields to aggregate by
    :param dest: path to the output normalized plate location
    :return: normalized plate as pd.Dataframe
    """
    # Calculate destination if not given (for compatibility reasons)
    if dest is None:
        if raw:
            if inter_channel:
                dest = 'raw'
            else:
                dest = 'raw1to1'
        else:
            dest = 'err'

        dest = f'{pure_fld}/{dest}/{os.path.basename(plate_csv)}'

    # Load output if already exist
    if os.path.exists(dest):
        index_size = len(well_index) if by_well else len(index_fields)
        return pd.read_csv(dest, index_col=list(range(index_size)))

    # Load original plate
    df = load_plate_csv(plate_csv, index_fields=index_fields)

    # Aggregate by well if needed
    if by_well:
        if well_index is None:
            well_index = ['Plate', LABEL_FIELD, 'Metadata_broad_sample', 'Image_Metadata_Well']

        df = df.groupby(by=well_index).apply(lambda g: g.mean())

    # Select the mock wells to learn their distribution
    df_mock = df[df.index.isin(['mock'], 1)]
    if not df_mock.shape[0]:
        print(f'no mock wells in {os.path.basename(plate_csv)}')
        return None

    # Learn the mock wells distribution
    scaler = StandardScaler()
    scaler.fit(df_mock)
    del df_mock

    # Normalize the entire plate
    df_zscores = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
    del df

    # Save the normalized plate at destination
    df_zscores.to_csv(dest)

    return df_zscores


def extract_z_score(plate_csv, by_well=True, by_channel=True, abs_zscore=True, well_type='treated', raw=False,
                    inter_channel=True):
    """
    Calculate the mean Z-Score measure per treatment's well
    :param plate_csv: path to a plate's csv
    :param by_well: whether to aggregate by well
    :param by_channel: whether to measure the aggregated score for each channel
    :param abs_zscore: whether to apply absolute to the raw z-scores
    :param well_type: list of well types to measure
    :param raw: whether it's raw features file or prediction (for load_pure_zscores)
    :param inter_channel: if raw=True, whether its raw features or predictions of 1to1 (for load_pure_zscores)
    :return: scores as pd.Dataframe
    """
    df = load_pure_zscores(plate_csv, raw, by_well, inter_channel)

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
    """
    Calculate the Frac-Score measure per treatment's well - ratio of features that above the threshold
    :param plate_csv: path to a plate's csv
    :param by_well: whether to aggregate by well
    :param by_channel: whether to measure the aggregated score for each channel
    :param abs_zscore: whether to apply absolute to the raw z-scores
    :param well_type: list of well types to measure
    :param raw: whether it's raw features file or prediction (for load_pure_zscores)
    :param thresh: threshold to identify which feature has a signal
    :param inter_channel: if raw=True, whether its raw features or predictions of 1to1 (for load_pure_zscores)
    :return: scores as pd.Dataframe
    """
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


def extract_ss_score_for_compound(cpdf, abs_zscore=True, th_range=range(2, 21)):
    """
    Calculate Signature Strength for a single compound
    :param cpdf: A single compound, each raw is a different well replicate
    :param abs_zscore: whether to apply absolute to the raw z-scores
    :param th_range: list of signature threshold to measure
    :return: pd.Series containing median correlation and SS&MAS per threshold in th_range
    """
    res = {}
    rep_cnt, fet_cnt = cpdf.shape

    corr = cpdf.astype('float64').T.corr(method='pearson').values
    if len(corr) == 1:
        med_corr = 1
    else:
        med_corr = np.median(list(corr[np.triu_indices(len(corr), k=1)]))

    res['Med_Corr'] = med_corr

    cpdf_norm = cpdf * np.sqrt(rep_cnt)

    if abs_zscore:
        cpdf_norm = abs(cpdf_norm.T)

    for t in th_range:
        gtr_t_cnt = (cpdf_norm >= t).sum().sum()
        ss_norm = gtr_t_cnt / rep_cnt
        mas = np.sqrt((max(med_corr, 0) * ss_norm) / fet_cnt)
        res[f'SS_{t}'] = ss_norm
        res[f'MAS_{t}'] = mas

    return pd.Series(res)


def extract_new_score_for_compound(cpdf, abs_zscore=True, th_range=range(2, 21), sqrt_norm=True, max_rep=None, n=1,
                                   value='median'):
    """
    Calculate adjusted Signature Strength for a single compound
    :param cpdf: A single compound, each raw is a different well replicate
    :param abs_zscore: whether to apply absolute to the raw z-scores
    :param th_range: list of signature threshold to measure
    :param sqrt_norm: whether to apply the square root adjustment
    :param max_rep: maximum number of replication allowed, sample if the given compound has more replicates
    :param n: number of times to repeat the calculations, useful when sampling
    :param value: how to aggregate the replicates, one of 'mean' and 'median'
    :return: pd.Dataframe containing replicates count and SS&MAS per threshold in th_range times n
    """
    res_cols = ['Rep_Cnt', *sum([[f'SS_{t}'] for t in th_range], [])]
    res = []

    rep_cnt, _ = cpdf.shape
    if max_rep and max_rep >= rep_cnt:
        n = 1

    for _ in range(n):
        cur_res = []

        cur_cpdf = cpdf
        rep_cnt, fet_cnt = cur_cpdf.shape
        if max_rep and max_rep < rep_cnt:
            cur_cpdf = cpdf.sample(max_rep)
            rep_cnt = max_rep

        cur_res.append(rep_cnt)

        cpdf_norm = cur_cpdf
        if abs_zscore:
            cpdf_norm = abs(cpdf_norm)

        if value == 'mean':
            cpd = cpdf_norm.mean()
        else:
            cpd = cpdf_norm.median()

        if sqrt_norm:
            cpd = cpd * np.sqrt(rep_cnt)

        for t in th_range:
            gtr_t_cnt = (cpd >= t).sum()
            ss_norm = gtr_t_cnt / rep_cnt
            # Add Normalizations for feature count
            ss_norm = ss_norm / fet_cnt
            cur_res.append(ss_norm)

        res.append(cur_res)

    return pd.DataFrame(res, columns=res_cols)


def extract_ss_score(df, expr_fld, th_range=[2, 6, 10, 14], cpd_id_fld='Metadata_broad_sample', new_ss=True,
                     value='mean', abs_zscore=False):
    """
    Calculate (adjusted) Signature Strength for a dataframe
    :param df: input pd.Dataframe
    :param expr_fld: experiment folder, where to save the output
    :param th_range: list of signature threshold to measure
    :param cpd_id_fld: field to group by - identify the compounds
    :param new_ss: whether to calculate the adjusted ss
    :param value: if new_ss, how to aggregate the replicates, one of 'mean' and 'median'
    :param abs_zscore: whether to apply absolute to the raw z-scores
    :return: None, the output is saved at expr_fld
    """
    if new_ss:
        print(f'calc with {value}')
        cur_res = df.groupby(cpd_id_fld).apply(extract_new_score_for_compound, abs_zscore=abs_zscore,
                                               th_range=th_range, value=value)
        cur_res.to_csv(os.path.join(expr_fld, f'ss-new-scores-{value}.csv'))
        print(f'saved to {expr_fld}')
    else:
        cur_res = df.groupby(cpd_id_fld).apply(extract_ss_score_for_compound, abs_zscore=abs_zscore,
                                               th_range=th_range)
        cur_res.to_csv(os.path.join(expr_fld, f'ss-scores.csv'))

    del df
    del cur_res


def extract_dist_score(plate_csv, well_type='treated', **kwargs):
    """
    Calculate a score based on Euclidean distance with no normalization
    :param plate_csv: path to a plate's csv
    :param well_type: list of well types to measure
    :param kwargs: not relevant
    :return: pd.Dataframe with the scores
    """
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
    """
    Calculate a score based on Euclidean distance while first normalizing the plate
    :param plate_csv: path to a plate's csv
    :param well_type: list of well types to measure
    :param kwargs:
    :return: pd.Dataframe with the scores
    """
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
    """
    Calculate a score based on Euclidean distance but normalizing the output distances based on mock
    :param plate_csv: path to a plate's csv
    :param well_type: list of well types to measure
    :param kwargs:
    :return: pd.Dataframe with the scores
    """
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
    """
    Calculate measures by using score_func (for compatibility reasons)
    :param score_func: score function from this file
    :param plate_csv: path to a plate's csv
    :param by_well: whether to aggregate by well
    :param well_type: list of well types to measure
    :param threshold: threshold to identify which feature has a signal
    :return: joined result for raw, 1to1 and 4to1
    """
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
    """
    Calculate measures by using score_func for all plates (for compatibility reasons)
    :param score_func: score function from this file
    :param by_well: whether to aggregate by well
    :param well_type: list of well types to measure
    :param threshold: threshold to identify which feature has a signal
    :return: joined result for raw, 1to1 and 4to1 for all plates
    """
    p = Pool(3)

    plates = [f[1] for f in files]
    score_results = p.starmap(extract_raw_and_err,
                              zip(cycle([score_func]), plates, cycle([by_well]), cycle([well_type]),
                                  cycle([threshold])))
    p.close()
    p.join()

    scores = {plate_number: score_results[i] for i, plate_number in enumerate([f[0] for f in files])}
    return scores


"""
The main of metrics.py
    -p : will run basic normalization per plate
         parallelling by plate (from an entire experiment folder)

    -s : will run Signature Strength calculations per channel
         parallelling by channel (from an entire experiment folder)
"""
if __name__ == '__main__':
    print('metrics main')
    print("Usage: metrics.py -p [experiment_path] [plate_index]")
    print("Usage: metrics.py -s [experiment_path] [channel_index]")
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

    elif sys.argv[1] == '-s':  # Means to extract ss scores
        exp_fld = sys.argv[2]
        channels = glob(os.path.join(exp_fld, '*'))
        # try:
        channel_idx = int(sys.argv[3])
        cur_fld = channels[channel_idx]
        plates = glob(os.path.join(cur_fld, 'zscores', '*'))
        cur_df = pd.concat([pd.read_csv(pth, index_col=[0, 1, 2, 3]) for pth in plates])
        cur_df = cur_df.query('Metadata_ASSAY_WELL_ROLE == "treated"').droplevel(1)
        print(f'start running for {cur_fld}')
        extract_ss_score(cur_df, cur_fld, th_range=range(2, 21), cpd_id_fld='Metadata_broad_sample', new_ss=True,
                         value='mean')
        # except:
        #     print(f'Error while reading channel {sys.argv[3]}')

    # For Visual Results
    # idx_fld = ['Plate', 'Well_Role', 'Broad_Sample', 'Well', 'Site', 'ImageNumber']
    # wll_idx = ['Plate', 'Well_Role', 'Broad_Sample', 'Well']
