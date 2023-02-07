import os
import pickle
import sys
from glob import glob
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def get_median_correlation(df, is_binary=False):
    if df.shape[0] == 1:
        return 1

    if is_binary:
        acc_scores = [accuracy_score(row1, row2) for ((_, row1), (_, row2)) in
                      combinations(df.astype('float64').iterrows(), 2)]
        return np.median(acc_scores)

    cor_mat = df.astype('float64').T.corr(method='pearson').values
    cor_mat = np.nan_to_num(cor_mat)

    if len(cor_mat) == 1:
        median_val = 1
    else:
        median_val = np.median(cor_mat[np.triu_indices(len(cor_mat), k=1)])

    return median_val


if __name__ == '__main__':
    null_dist_path, slice_size, slice_id, dest = sys.argv[1:]
    slice_size = int(slice_size)
    slice_id = int(slice_id)
    binarize = True
    t = 2

    zscores = {
        # '4to1':{'path':f'/storage/users/g-and-n/tabular_models_results/41/ALL/zscores'},
        # '2to1':{'path':f'/storage/users/g-and-n/tabular_models_results/21/ALL/zscores'},
        '1to1': {'path': f'/storage/users/g-and-n/tabular_models_results/111/ALL/zscores'},
        'raw': {'path': f'/storage/users/g-and-n/tabular_models_results/30000/results/z_scores/pure/raw'},
        # '5to5':{'path':f'/storage/users/g-and-n/tabular_models_results/55/ALL/zscores'},
    }

    # get plate numbers
    cur_fld = f'/storage/users/g-and-n/tabular_models_results/41/ALL/'
    all_plates = glob(os.path.join(cur_fld, 'zscores', '*'))
    plate_nums = [os.path.split(plate)[-1] for plate in all_plates]

    # load zscores
    for model in zscores.keys():
        plates = [os.path.join(zscores[model]['path'], plate) for plate in plate_nums]
        zscores[model]['all'] = pd.concat([pd.read_csv(pth, index_col=[0, 1, 2, 3]) for pth in plates])

        # Binarize
        if binarize:
            if model == 'raw':
                zscores[model]['all'] = zscores[model]['all'].abs()
            zscores[model]['all'] = zscores[model]['all'].apply(lambda s: s.apply(lambda x: 1.0 if x >= t else 0.0))

    # Drop columns that are not in other results
    # find joint columns
    joint_cols = []
    for model in zscores.keys():
        model_cols = zscores[model]['all'].columns
        if len(joint_cols) == 0:
            joint_cols = model_cols
        else:
            joint_cols = np.intersect1d(joint_cols, model_cols)

    # drop non-joint columns
    for model in zscores.keys():
        zscores[model]['all'].drop(columns=[c for c in zscores[model]['all'].columns if c not in joint_cols],
                                   inplace=True)
        print(zscores[model]['all'].shape)

    for model in zscores.keys():
        zscores[model]['trt'] = zscores[model]['all'].query('Metadata_ASSAY_WELL_ROLE == "treated"')
        # zscores[model]['ctl'] = zscores[model]['all'].query('Metadata_ASSAY_WELL_ROLE == "mock"')

    with open(null_dist_path, 'rb') as f:
        # Load the pickle file
        null_distribution_replicates = pickle.load(f)

    cpds = list(null_distribution_replicates.keys())
    cpds.sort()
    cpds = cpds[slice_id * slice_size:(slice_id + 1) * slice_size]

    for model in zscores.keys():
        print(5, model)
        cur_dest = os.path.join(dest, model)
        os.makedirs(cur_dest, exist_ok=True)
        cur_dest = os.path.join(cur_dest, f'{slice_id}.pickle')
        res = {}
        for cpd in cpds:
            print(cpd)
            cpd_replicates = zscores[model]['trt'][zscores[model]['trt'].index.isin([cpd], 2)].copy()
            cpd_med_score = get_median_correlation(cpd_replicates, binarize)
            del cpd_replicates

            null_dist = null_distribution_replicates[cpd]
            null_dist_scores = []
            for null_idxs in null_dist:
                cur_null = zscores[model]['trt'][zscores[model]['trt'].index.isin(null_idxs)].copy()
                curr_null_score = get_median_correlation(cur_null, binarize)
                null_dist_scores.append(curr_null_score)
                del cur_null

            res[cpd] = (cpd_med_score, null_dist_scores)

        with open(cur_dest, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
