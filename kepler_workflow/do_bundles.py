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
    names,
    quarter="all",
    version="1.1.1",
    return_lkf=False,
    out_dir="./lcs",
    save_file_names=False,
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
    else:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

    if quarter == "all":
        quarter = np.arange(0, 18)
    else:
        quarter = list(quarter)

    if save_file_names:
        for tarname in tqdm(id4):
            names_in = []
            for name in names:
                if name.startswith(tarname):
                    names_in.append(f"{tarname}/{name}")
                    # names_in.append(
                    #     f"{tarname}/{name}/hlsp_kbonus-bkg_kepler_kepler_kic-"
                    #     f"{name}-q*_kepler_v{version}_lc.fits"
                    # )
            cmd = (
                f"tar -xv --directory /nobackup/jimartin/ADAP/kbonus/lcs/kepler"
                f"/{os.path.basename(out_dir)} -f {tarname}.tar {' '.join(names_in)}"
            )
            np.savetxt(
                f"lcs_in_{tarname}_{os.path.basename(out_dir)}.sh",
                np.array([cmd]),
                fmt="%s",
            )
    else:
        for tarname in tqdm(id4, desc="Tar files"):
            tarpath = f"{LCS_PATH}/kepler/{tarname}.tar"
            print(f"Unpacking from {tarpath}")

            with tarfile.open(tarpath, "r") as tar:
                for name in tqdm(names, leave=True):
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


def do_bundle(targets="wd", version="1.1.1", save_file_names=False):

    if targets == "wd":
        fname = f"{PACKAGEDIR}/data/catalogs/tpf/kbonus-bkg_kepler_v{version}_source_catalog_wd.csv"
    elif targets == "mstars":
        fname = f"{PACKAGEDIR}/data/catalogs/tpf/kbonus-bkg_kepler_v{version}_source_catalog_mstars.csv"
    else:
        raise ("Wong targets")

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
        out_dir=f"{LCS_PATH}/kepler/{targets}",
        save_file_names=save_file_names,
    )

    return
