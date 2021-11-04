import os
import tarfile
import numpy as np
from tqdm import tqdm

from paths import ARCHIVE_PATH, OUTPUT_PATH, LCS_PATH, PACKAGEDIR
from make_lightcurves import get_file_list

def main():

    file_list = get_file_list(1, 9, -1, 1, tar_tpfs=True)

    out_dir = f"{PACKAGEDIR}/download/"
    if not os.path.isdir(out_dir):
        os.makedirs(f"{PACKAGEDIR}/download/")

    for k, fname in tqdm(enumerate(fname_list), total=len(file_list)):
        tarf = f"{fname.split('/')[0]}_{fname.split('/')[1]}.tar"
        tarf = f"{ARCHIVE_PATH}/data/kepler/tpf/{fname.split('/')[0]}/{tarf}"
        tarfile.open(tarf, mode="r").extract(fname, out_dir)


if __name__ == '__main__':
    main()
    print("Done!")
