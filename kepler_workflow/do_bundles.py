import os
import sys
import tarfile
import tempfile
import numpy as np
import pandas as pd
import lightkurve as lk
from glob import glob
from tqdm import tqdm

from paths import *


def get_lcs_from_archive(
    names, quarter="all", version="1.1.1", return_lkf=False, out_dir="./lcs"
):
    if isinstance(names, (int, str)):
        names = [names]
    if isinstance(names[0], int):
        names = [f"{x:09}" if x < 22934493 else str(x) for x in names]

    id4 = list(set([x[:4] for x in names]))
    if return_lkf:
        tmpdir = tempfile.TemporaryDirectory(prefix="temp_fits")
        out_dir = tmpdir.name
        lks = []

    if quarter == "all":
        quarter = np.arange(0, 18)
    else:
        quarter = list(quarter)

    for tarname in id4:
        tarpath = f"{LCS_PATH}/kepler/{tarname}.tar"
        print(f"Unpacking from {tarpath}")

        with tarfile.open(tarpath, "r") as tar:
            for name in tqdm(names):
                for q in quarter:
                    lc_name = (
                        f"{tarname}/{name}/hlsp_kbonus-bkg_kepler_kepler_kic-"
                        f"{name}-q{q:02}_kepler_v{version}_lc.fits"
                    )
                    try:
                        # if True:
                        tar.extract(lc_name, out_dir)
                    except KeyError:
                        continue
                    if return_lkf:
                        lks.append(lk.KeplerLightCurve.read(f"{out_dir}/{lc_name}"))

    if return_lkf:
        tmpdir.cleanup()
        return lk.LightCurveCollection(lks)
    else:
        return


def do_bundle(version="1.1.1"):

    fname = f"{PACKAGEDIR}/data/catalogs/tpf/kbonus-bkg_kepler_v{version}_source_catalog_wd.csv"

    df = pd.read_csv(fname)

    names = []
    for k, row in df.iterrows():
        if row.kic < 1.000000e20:
            names.append(int(row.kic))
        else:
            names.append(int(row.gaia_designation.split(" ")[-1]))

    get_lcs_from_archive(
        names,
        quarter="all",
        version="1.1.1",
        return_lkf=False,
        out_dir=f"{LCS_PATH}/kepler/white_dwarfs",
    )

    return
