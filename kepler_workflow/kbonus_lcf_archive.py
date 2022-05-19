import os
import sys
import glob
import shutil
import tarfile
import argparse
import fitsio
import shutil
import tempfile
from tqdm import tqdm
import numpy as np

from paths import *


def do_archive(quarter, channel, suffix="fvaT_bkgT_augT_sgmT_iteT", version="1.1.1"):

    print(
        f"{LCS_PATH}/kepler/ch{channel:02}/q{quarter:02}/"
        f"kbonus-kepler-bkg_ch{channel:02}_q{quarter:02}"
        f"_v{version}_lcs_b*_{suffix}.tar.gz"
    )
    tar_files = sorted(
        glob.glob(
            f"{LCS_PATH}/kepler/ch{channel:02}/q{quarter:02}/"
            f"kbonus-kepler-bkg_ch{channel:02}_q{quarter:02}"
            f"_v{version}_lcs_b*_{suffix}.tar.gz"
        )
    )
    print(f"Total tar files: {len(tar_files)}")

    for i, tf in enumerate(tar_files):
        print(f"Working with file {i+1}/{len(tar_files)}: {tf}")
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(tf) as tar:
                members = tar.getmembers()
                labels = [x.name.split("/")[-1].split("_")[4][:-4] for x in members]

                ids = [
                    f"{int(x.split('-')[-1]):09}"
                    if x.startswith("KIC")
                    else x.split("-")[-1]
                    for x in labels
                ]
                dirs = list(set([x[:4] for x in ids]))
                print("Creating the following directories:")
                print(dirs)

                for dir in dirs:
                    dir_path = f"{LCS_PATH}/kepler/{dir}"
                    if not os.path.isdir(dir_path):
                        os.makedirs(dir_path)

                for k, member in tqdm(
                    enumerate(members),
                    total=len(members),
                    desc="Extracting FITS into archive",
                ):
                    dirout = f"{LCS_PATH}/kepler/{ids[k][:4]}/{ids[k]}"
                    if not os.path.isdir(dirout):
                        os.makedirs(dirout)
                    fout = f"{dirout}/{member.name}"
                    if os.path.isfile(fout):
                        tar.extract(member, path=tmpdir)
                        shutil.copy(
                            f"{tmpdir}/{member.name}",
                            fout.replace("_lc.fits", "_lc_2.fits"),
                        )
                    else:
                        tar.extract(member, path=dirout)


def drop_duplicates(dir):
    print(f"Working on {dir:04}")
    print(f"{LCS_PATH}/kepler/{dir:04}/*/hlsp_kbonus-kbkgd_kepler_kepler_*_lc_2.fits")
    dupfiles = glob.glob(
        f"{LCS_PATH}/kepler/{dir:04}/*/hlsp_kbonus-kbkgd_kepler_kepler_*_lc_2.fits"
    )
    if len(dupfiles) == 0:
        print("No duplicated files")
        return

    print(f"Total duplicated LCs {len(dupfiles)}")
    for sec in dupfiles:
        fir = sec.replace("_lc_2.fits", "_lc.fits")
        err_means = [
            fitsio.read(fir, ext=1, columns="FLUX_ERR").mean(),
            fitsio.read(sec, ext=1, columns="FLUX_ERR").mean(),
        ]
        if np.isnan(err_means).all():
            err_means = [
                fitsio.read(fir, ext=1, columns="SAP_FLUX_ERR").mean(),
                fitsio.read(sec, ext=1, columns="SAP_FLUX_ERR").mean(),
            ]
        if np.isnan(err_means).all():
            continue
        print(err_means)
        if (np.array(err_means) == 0).all():
            print("All empty lcs, removing both...")
            os.remove(fir)
            os.remove(sec)
        elif np.nanargmin(err_means) == 0:
            print("keep 1")
            os.remove(sec)
        elif np.nanargmin(err_means) == 1:
            print("keep 2")
            os.remove(fir)
            shutil.move(sec, fir)
        else:
            continue
        print("----" * 5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quarter",
        dest="quarter",
        type=int,
        default=None,
        help="Quarter number.",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        type=int,
        default=None,
        help="Channel number",
    )
    parser.add_argument(
        "--dir",
        dest="dir",
        type=int,
        default=None,
        help="Kepler 4-digit directory",
    )
    parser.add_argument(
        "--suffix",
        dest="suffix",
        type=str,
        default="fvaT_bkgT_augT_sgmT_iteT",
        help="File prefix",
    )
    args = parser.parse_args()
    if args.quarter and args.channel:
        do_archive(args.quarter, args.channel, suffix=args.suffix)

    if args.dir:
        drop_duplicates(args.dir)
