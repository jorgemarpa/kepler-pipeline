import os
import glob
import numpy as np
from paths import ARCHIVE_PATH, PACKAGEDIR

def run():
    folder_list = np.sort(glob.glob(f"{ARCHIVE_PATH}/kepler/tpfs/*"))
    print(len(folder_list))

    for folder in folder_list:
        fname = folder.split('/')[-1]
        table_list = np.sort(glob.glob(f"{PACKAGEDIR}/data/support/kepler_tpf_map_{fname}_q*_tar.csv"))
        print(f"Folder name {fname} totla tables {len(table_list)}")

if __name__ == '__main__':
    run()
