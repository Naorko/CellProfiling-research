import os
import random
import sys

import pandas as pd


def sample_split_plate(plate, by_field, sample_n):
    values = set(plate[by_field].unique().tolist())
    sample = set(random.sample(values, sample_n)) if sample_n < len(values) else values.copy()
    return sample, values - sample


def create_tabular_metadata(plates_path, plates, label_field, train_labels, by_fld, sample_n):
    mt_dict = {'Plate': [], label_field: [], 'Mode': [], by_fld: [], 'Count': []}
    for plate in plates:
        plate_path = os.path.join(plates_path, f'{plate}.csv')
        df = pd.read_csv(plate_path)
        for lbl in df[label_field].unique():
            qdf = df.query(f'{label_field} == "{lbl}"')
            if lbl in train_labels:
                train_set, test_set = sample_split_plate(qdf, by_fld, sample_n)
                train_count = qdf[by_fld].isin(train_set).sum()
                test_count = qdf.shape[0] - train_count
                add_metadata(mt_dict, label_field, by_fld, plate, lbl, 'train', train_set, train_count)
            else:
                test_set = set(qdf[by_fld].unique())
                test_count = qdf.shape[0]

            if test_count:
                add_metadata(mt_dict, label_field, by_fld, plate, lbl, 'predict', test_set, test_count)
        del qdf, df

    return pd.DataFrame(mt_dict)


def add_metadata(mt_dict, label_field, by_fld, plate, lbl, mode, train_set, train_count):
    mt_dict['Plate'].append(plate)
    mt_dict[label_field].append(lbl)
    mt_dict['Mode'].append(mode)
    mt_dict[by_fld].append(train_set)
    mt_dict['Count'].append(train_count)


if __name__ == '__main__':
    plates_fld = '/storage/users/g-and-n/plates/csvs/'
    output_fld = '/storage/users/g-and-n/plates/tabular_metadata/'
    by_fld = 'Image_Metadata_Well'
    sample_n = 16
    label_field = 'Metadata_ASSAY_WELL_ROLE'
    train_labels = ['mock']

    i = int(sys.argv[1])
    # i=0
    plates = [p.split('.')[0] for p in os.listdir(plates_fld)][i * 6:i * 6 + 6]

    mt_df = create_tabular_metadata(plates_fld, plates, label_field, train_labels, by_fld, sample_n)
    mt_df.loc[mt_df['Metadata_ASSAY_WELL_ROLE'].isna(), 'Metadata_ASSAY_WELL_ROLE'] = 'mock'
    mt_df.to_csv(os.path.join(output_fld, f'{i}.csv'), index=False)
