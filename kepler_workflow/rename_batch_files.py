import os
import sys
from glob import glob
import numpy as np
from tqdm import tqdm
from paths import ARCHIVE_PATH, OUTPUT_PATH, LCS_PATH, PACKAGEDIR

quarter = 2
channels = np.arange(0, 18)
suffix = "fvaT_bkgF_augT_sgmF_iteT_cbvT"
version = "1.1.1"

for ch in channels:
    in_files = sorted(
        glob(
            f"{LCS_PATH}/kepler/ch{ch:02}/q{quarter:02}/"
            f"kbonus-kepler-bkg_ch{ch:02}_q{quarter:02}_"
            f"v{version}*_{suffix}.*"
        )
    )
    print(f"Total files {len(in_files)}")
    for fi in tqdm(in_files, total=len(in_files)):
        fo = fi.replace("bkgF", "bkgT")
        # print(f"{fi} -> {fo}")
        os.rename(fi, fo)
    print("Done!")
