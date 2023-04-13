import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

use_cuda = torch.cuda.is_available()
print("USE CUDA=" + str(use_cuda))
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

DTYPE = f'float32'


class TabularDataset(Dataset):
    """
    Load and filter plates on ram
    """
    def __init__(self,
                 metadata_df,
                 root_dir,
                 input_fields,
                 target_fields=None,
                 index_fields=None,
                 target_channel=None,
                 transform=None,
                 for_data_statistics_calc=False,
                 is_test=False,
                 shuffle=True,
                 dfs_load_count=1,
                 max_load=2):

        super(Dataset).__init__()
        self.metadata_df = metadata_df
        metadata_df['CumCount'] = metadata_df['Count'].cumsum()
        self.root_dir = root_dir
        self.target_channel = target_channel.value if target_channel else None
        self.target_channel_enum = target_channel
        self.input_fields = input_fields
        self.target_fields = target_fields
        self.index_fields = index_fields
        self.transform = transform

        self.for_data_statistics_calc = for_data_statistics_calc
        self.is_test = is_test

        self.shuffle = shuffle

        # Load Params
        # self.dfs_load_count = dfs_load_count
        self.max_load = max_load
        self.loaded_dfs = []
        self.current_start_end = []
        self.meta_mapper = {}

    def __len__(self):
        """
        :return: The amount of rows in the dataset
        """
        return self.metadata_df['Count'].sum()

    def load_cell(self, index):
        """
        Returns the tabular data of an index
        :param index: cell index in metadata table
        :return: np.ndarray the tabular data of an index
        """
        # Retrieve the row from metadata and calculate the row index in the dataset
        p, lbl, mode, filter_set, c, cc = self.metadata_df[self.metadata_df['CumCount'] > index].iloc[0]
        idx = index - (cc - c)

        # Check if the row index already loaded
        found = False
        for i, (start, end) in enumerate(self.current_start_end):
            if start <= index <= end:
                found = True
                break

        # Row index not loaded - load new dataframe to ram
        if not found:
            # Remove old dataframe if cap reached
            if len(self.loaded_dfs) == self.max_load:
                del self.loaded_dfs[0]
                del self.current_start_end[0]

            # Load and filter according to metadata row
            plate_path = os.path.join(self.root_dir, f'{p}.csv')
            new_df = pd.read_csv(plate_path)
            # new_df.dropna(inplace=True)
            new_df.fillna(new_df[self.input_fields+self.target_fields].mean(), inplace=True)
            new_df = new_df.query(f'{self.metadata_df.columns[1]} == "{lbl}"')
            new_df = new_df[new_df[self.metadata_df.columns[3]].isin(filter_set)]

            # TODO: Remove unnecessary fields ?
            if self.shuffle:
                new_df = new_df.sample(frac=1)

            # Add new dataframe to the loaded dataframes
            self.loaded_dfs.append(new_df)
            start_idx = self.current_start_end[-1][1]+1 if self.current_start_end else 0
            end_idx = start_idx + c - 1
            self.current_start_end.append((start_idx, end_idx))
            i = -1

        # Get the cell row by the index
        row = self.loaded_dfs[i].iloc[idx]
        ind, data = row[self.index_fields], row[self.input_fields + self.target_fields]
        return ind.to_numpy().squeeze(), data.to_numpy().squeeze().astype(np.dtype(DTYPE))

    def __getitem__(self, idx):
        """
        Get row of the dataset by index
        :param idx:
        :return: row seperated as (input, target, index)
        """
        ind, inp = self.load_cell(idx)
        if not self.for_data_statistics_calc:
            if self.transform:
                inp = self.transform(inp)

            inp, target = inp[:len(self.input_fields)], inp[len(self.input_fields):]

            if self.is_test:
                self.meta_mapper[idx] = ind
                return inp, target, idx
            else:
                return inp, target
        else:
            return inp

    def get_index(self):
        """
        Get the index fields' values of the whole dataset
        :return: The index
        """
        index = pd.DataFrame(columns=self.index_fields)
        lbl_field = self.metadata_df.columns[1]
        for _, (plate, lbl, _, _) in self.metadata_df.iterrows():
            plate_path = os.path.join(self.root_dir, f'{plate}.csv')
            df = pd.read_csv(plate_path).query(f'{lbl_field} == "{lbl}"')
            index = index.append(df[self.index_fields], ignore_index=True)

        return index
