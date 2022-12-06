import os
import json
import random
import numpy as np
import pandas as pd

# taken from https://github.com/broadinstitute/lincs-profiling-complementarity/blob/master/1.Data-exploration/Consensus/cell_painting/1.cellpainting_moa_median_scores_calculation.ipynb

from pycytominer import feature_select


def feature_selection(data):
    """
    Perform feature selection by dropping columns with null or
    only zeros values, and highly correlated values from the data.

    params:
    dataset_link: string of github link to the consensus dataset

    Returns:
    data: returned consensus dataframe

    """
    # data = pd.read_csv(dataset_link, compression='gzip', error_bad_lines=False)
    cols = data.columns.tolist()
    has_nulls = [x for x in cols if data[x].isnull().sum()]
    all_zeros = [x for x in cols if all(y == 0.0 for y in data[x].values)]
    # data.drop(has_nulls+all_zeros, axis=1, inplace=True)
    data.dropna(axis=0, inplace=True)
    excluded_cols = feature_select(
        data,
        operation=[
            "correlation_threshold",  # Exclude features that have correlations above a certain threshold (default 0.9)
            "variance_threshold"  # Exclude features that have low variance (low information content)
        ],
        # blocklist_file="https://raw.githubusercontent.com/broadinstitute/lincs-cell-painting/1769b32c7cef3385ccc4cea7057386e8a1bde39a/utils/consensus_blocklist.txt"
    )
    return {'has_nulls': has_nulls, 'all_zeros': all_zeros, **excluded_cols}


plates_pth = '/sise/assafzar-group/g-and-n/plates/csvs'
plates = os.listdir(plates_pth)

cols_pth = '/sise/assafzar-group/g-and-n/plates/columns.txt'
with open(cols_pth, 'r') as f:
    meta_cols = json.load(f)

del meta_cols['MIXED']
meta_cols = {
    'ALL': sum([cols for k, cols in meta_cols.items() if k != 'GENERAL'], [])
}

processed = []
cur_df = None


def add_plate(plate):
    global cur_df
    df = pd.read_csv(os.path.join(plates_pth, plate), index_col=(list(range(6))))
    if cur_df is None:
        cur_df = df
    else:
        cur_df = pd.concat([cur_df, df])
    del df


# for i in range(23):
#     cur_plate = plates.pop(random.randrange(len(plates)))
#     processed.append(cur_plate)
#     add_plate(cur_plate)

while plates:
    cur_plate = plates.pop(random.randrange(len(plates)))
    processed.append(cur_plate)
    print(f'Try adding plate {cur_plate} to have {len(processed)} plates')

    add_plate(cur_plate)

    for chan, chan_cols in meta_cols.items():
        cols = feature_selection(cur_df[chan_cols].copy())
        with open(f'/sise/assafzar-group/g-and-n/feature_selection/{chan}_{len(processed)}.sav', 'w') as f:
            out = {'plates': processed, 'cols': cols}
            json.dump(out, f)
            del cols
