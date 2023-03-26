import os
from tqdm import tqdm

import pandas as pd

plates_dir = '/storage/users/g-and-n/plates/csvs/'
index_fields = ['Plate', 'Metadata_ASSAY_WELL_ROLE', 'Metadata_broad_sample',
                'Image_Metadata_Well', 'ImageNumber', 'ObjectNumber']
extract_path = '/storage/users/g-and-n/plates/tabular_indexes/'
os.makedirs(extract_path, exist_ok=True)

plates = os.listdir(plates_dir)
index = pd.DataFrame(index_fields)
for plate in tqdm(plates):
    plate_path = os.path.join(plates_dir, plate)
    df = pd.read_csv(plate_path)
    index_path = os.path.join(extract_path, plate)
    df[index_fields].to_csv(index_path, index=False)
    del df
