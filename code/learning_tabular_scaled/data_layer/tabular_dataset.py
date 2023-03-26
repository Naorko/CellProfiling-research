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

# Pixel Attributes
BITS_PER_PIXEL = 16  # 16 for tiff, 8 for png
VMAX = 2 ** BITS_PER_PIXEL - 1
DTYPE = f'float32'


class TabularDataset(Dataset):
    def __init__(self,
                 metadata_df,
                 root_dir,
                 input_fields,
                 target_fields=None,
                 index_fields=None,
                 target_channel=None,
                 transform=None,
                 for_data_statistics_calc=False,
                 is_test=False):

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

    def __len__(self):
        return self.metadata_df['Count'].sum()

    def load_cell(self, index):
        """
        Returns the tabular data of an index
        Parameters
        ----------
        index : int
            cell index in metadata table
        Returns
        -------
        np.ndarray the tabular data of an index
        """

        p, lbl, idxs, c, cc = self.metadata_df[self.metadata_df['CumCount'] > index].iloc[0]
        index_idx = index - (cc - c)
        row_idx = idxs[index_idx] + 1
        plate_path = os.path.join(self.root_dir, f'{p}.csv')
        fields_names = pd.read_csv(plate_path, nrows=0).columns
        row = pd.read_csv(plate_path, nrows=1, skiprows=row_idx, names=fields_names)
        ind, data = row[self.index_fields], row[self.input_fields + self.target_fields]
        return ind.to_numpy().squeeze(), data.to_numpy().squeeze().astype(np.dtype(DTYPE))

    def __getitem__(self, idx):
        ind, inp = self.load_cell(idx)
        if not self.for_data_statistics_calc:
            if self.transform:
                inp = self.transform(inp)

            inp, target = inp[:len(self.input_fields)], inp[len(self.input_fields):]

            if self.is_test:
                return inp, target, ind
            else:
                return inp, target
        else:
            return inp
