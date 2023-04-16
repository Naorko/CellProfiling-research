import os
"""
Some constants used by metrics.py, for compatibility reasons
"""

# Tabular
model = 'DNN'
# err_fld = f'/storage/users/g-and-n/tabular_models_results/30000/results/errors/'
raw_fld = '/storage/users/g-and-n/plates/csvs'
# raw1to1_fld = f'/storage/users/g-and-n/tabular_models_results/30101/results/errors'
# zsc_fld = f'/storage/users/g-and-n/tabular_models_results/30000/results/z_scores'
# pure_fld = f'{zsc_fld}/pure'

# Visual

# model = 'DNN'
err_fld = f'/storage/users/g-and-n/visual_models_results/30000/UNET4TO1/results/errors'
raw1to1_fld = f'/storage/users/g-and-n/visual_models_results/30000/AUTO1TO1/results/errors'
zsc_fld = f'/storage/users/g-and-n/visual_models_results/30000/results/z_scores'
pure_fld = f'{zsc_fld}/pure'

# plots_path = f'/home/naorko/CellProfiling/plots/fraction-score/{model}'

files = [(int(f.name.split('.')[0]), f.name) for f in os.scandir(err_fld) if f.name.endswith('.csv')]
