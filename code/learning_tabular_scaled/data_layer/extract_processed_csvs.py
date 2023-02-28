import os
import sys

sys.path.append(os.path.abspath('../..'))

from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from learning_tabular_scaled.configuration.config import parse_args


def main(metadata_df, root_dir, out_dir,normalize_by_plate=False):
    split_field = metadata_df.columns[3]
    metadata_df[split_field] = metadata_df[split_field].apply(eval)

    # mins = defaultdict(list)
    means = defaultdict(list)
    stds = defaultdict(list)

    # normalize to min max by
    if normalize_by_plate:
        metadata_df_train = metadata_df.loc[metadata_df['Mode']=='train',:]
        for _, (p, lbl, mode, filter_set, c) in tqdm(metadata_df_train.iterrows()):

            out_path = os.path.join(out_dir, f'{p}_{lbl}_{mode}.csv')
            if not os.path.exists(out_path):
                plate_path = os.path.join(root_dir, f'{p}.csv')
                new_df = pd.read_csv(plate_path)
                new_df.fillna(new_df.mean(), inplace=True)
                new_df = new_df[new_df[split_field].isin(filter_set)]

                means[f'{p}'] = new_df.mean()
                # maxs[f'{p}'] = new_df.max()
                stds[f'{p}'] =  new_df.std()


    for _, (p, lbl, mode, filter_set, c) in tqdm(metadata_df.iterrows()):

        out_path = os.path.join(out_dir, f'{p}_{lbl}_{mode}.csv')
        if not os.path.exists(out_path):
            plate_path = os.path.join(root_dir, f'{p}.csv')
            new_df = pd.read_csv(plate_path)
            # new_df.dropna(inplace=True)
            new_df.fillna(new_df.mean(), inplace=True)
            if normalize_by_plate:
                new_df.subtract(means[f'{p}']).div(stds[f'{p}'])
            # new_df = new_df.query(f'{metadata_df.columns[1]} == "{lbl}"')
            new_df = new_df[new_df[split_field].isin(filter_set)]
            new_df.to_csv(out_path, index=False)


if __name__ == '__main__':
    i =int(sys.argv[1])
    # inp = 1

    channels = ['AGP', 'DNA', 'ER', 'Mito', 'RNA']
    exp_params = [
        # (['AGP'], ['Mito', 'RNA'], 16),
        # (['DNA'], ['AGP', 'Mito'], 16),
        # (['ER'], ['AGP', 'RNA'], 16),
        # (['Mito'], ['AGP', 'ER'], 16),
        # (['RNA'], ['AGP', 'ER'], 16),
        *[([chan], [chan], 8) for chan in channels],
        # *[([chan], channels[:i] + channels[i + 1:], 16) for i, chan in enumerate(channels)],
        # (['AGP', 'DNA', 'ER', 'Mito', 'RNA'], ['AGP', 'DNA', 'ER', 'Mito', 'RNA'], 8)
    ]

    exp_num = 50001
    out_channels, in_channels, lsd = exp_params[inp]
    # if exp_num != 11:
    #     in_channels = ['GENERAL'] + in_channels
    args = parse_args(exp_num=exp_num, in_channels=in_channels, out_channels=out_channels)

    normalize_by_plate=True
    # i=0
    mt_pth = '/storage/users/g-and-n/plates/tabular_metadata.csv'
    r_dir = '/storage/users/g-and-n/plates/csvs/'
    if normalize_by_plate:
        o_dir = '/storage/users/g-and-n/plates/csvs_processed_normalized/'
    else:
        o_dir = '/storage/users/g-and-n/plates/csvs_processed/'
    os.makedirs(o_dir,exist_ok=True)
    metadata_df = pd.read_csv(mt_pth)
    main(metadata_df.iloc[i*5:(i+1)*5], r_dir, o_dir,normalize_by_plate)
