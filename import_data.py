import glob
import os
import pandas as pd
import numpy as np

# Data folder.
data_path = 'data\\patch_LHCP'

# Files list.
files_list = glob.glob(os.path.join(data_path, 'source_data\\*.txt'))

for i in range(0, len(files_list)):
    raw_data = pd.read_csv(files_list[i], sep='\s+', dtype=str, usecols=(0, 1, 3, 4, 5, 6), header=None).values
    raw_data = np.delete(raw_data, (0, 1), 0)
    raw_data = raw_data.astype(float)
    np.savetxt(os.path.join(data_path, os.path.basename(files_list[i])), raw_data)

# Phase centers.
# phase_centers = np.array(pd.read_excel(os.path.join(data_path, 'source_data\\phase_centers.xlsx'), header=None))/1000
# np.savetxt(os.path.join(data_path, 'phase_centers.txt'), phase_centers)