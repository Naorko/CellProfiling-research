import os
import pandas as pd
import numpy as np
from glob import glob
import sys

inp = int(sys.argv[1])
i = inp // 2
model_id = inp % 2

model = ['AUTO1TO1', 'UNET4TO1'][model_id]

os.chdir(f'/storage/users/g-and-n/visual_models_results/30000/{model}')
res_path = f'/storage/users/g-and-n/visual_models_results/30000/{model}/results/errors'
os.makedirs(res_path, exist_ok=True)

channels = glob('./channel */results/')
channels_name = [c.split(' ')[1] for c in glob('./channel *')]

plates = [os.path.basename(r) for r in glob('./channel AGP/results/*')]
plates.sort()

df_index = ['Plate', 'Well', 'Site', 'ImageNumber', 'Well_Role', 'Broad_Sample']
# df_index = ['Plate', 'Metadata_ASSAY_WELL_ROLE', 'Metadata_broad_sample',
#             'Image_Metadata_Well', 'ImageNumber', 'ObjectNumber']


def merge(p):
    dfs = [pd.read_csv(os.path.join(c, p)) for c in channels]

    _ = [df.set_index(df_index, inplace=True) for df in dfs]
    dfs = [df.add_prefix(f'{c}_') for c, df in zip(channels_name, dfs)]
    df = pd.concat(dfs, axis=1)

    pcc_all = df.filter(regex='_PCC', axis=1).mean(axis=1)
    pcc_all.name = 'ALL_PCC'
    mse_all = df.filter(regex='_MSE', axis=1).mean(axis=1)
    mse_all.name = 'ALL_MSE'

    df = pd.concat([df, pcc_all, mse_all], axis=1)

    df.to_csv(os.path.join(res_path, p))


merge(plates[i])
