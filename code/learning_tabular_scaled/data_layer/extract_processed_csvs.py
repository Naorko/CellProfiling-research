import os
import sys
import pandas as pd
from tqdm import tqdm


def main(metadata_df, root_dir, out_dir):
    split_field = metadata_df.columns[3]
    metadata_df[split_field] = metadata_df[split_field].apply(eval)
    for _, (p, lbl, mode, filter_set, c) in tqdm(metadata_df.iterrows()):
        out_path = os.path.join(out_dir, f'{p}_{lbl}_{mode}.csv')
        if not os.path.exists(out_path):
            plate_path = os.path.join(root_dir, f'{p}.csv')
            new_df = pd.read_csv(plate_path)
            # new_df.dropna(inplace=True)
            new_df.fillna(new_df.mean(), inplace=True)
            # new_df = new_df.query(f'{metadata_df.columns[1]} == "{lbl}"')
            new_df = new_df[new_df[split_field].isin(filter_set)]
            new_df.to_csv(out_path, index=False)


if __name__ == '__main__':
    i =int(sys.argv[1])
    mt_pth = '/storage/users/g-and-n/plates/tabular_metadata.csv'
    r_dir = '/storage/users/g-and-n/plates/csvs/'
    o_dir = '/storage/users/g-and-n/plates/csvs_processed/'

    metadata_df = pd.read_csv(mt_pth)
    main(metadata_df.iloc[i*5:(i+1)*5], r_dir, o_dir)
