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
import pandas as pd
from astropy.io import fits

from paths import *


def do_archive(
    quarter,
    channel,
    suffix="fvaT_bkgT_augT_sgmT_iteT",
    version="1.1.1",
):

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
    print(f"{LCS_PATH}/kepler/{dir:04}/*/hlsp_kbonus-bkg_kepler_kepler_*_lc_2.fits")
    dupfiles = glob.glob(
        f"{LCS_PATH}/kepler/{dir:04}/*/hlsp_kbonus-bkg_kepler_kepler_*_lc_2.fits"
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


def make_tarball_archive(folders=None, version="1.1.1", delete=False):

    if folders is None:
        folders = sorted(glob.glob(f"{LCS_PATH}/kepler/*"))
        folders = [x for x in folders if os.path.isdir(x)]
        folders = [x for x in folders if "ch" not in os.path.basename(x)]
        folders = [
            x for x in folders if os.path.basename(x) not in ["bkp", "tmp", "download"]
        ]
    else:
        folders = [f"{folders:04}"]

    print(f"Creating tarball files for {len(folders)} folders...")
    for dir in tqdm(folders, total=len(folders), desc="Folder"):
        id4 = os.path.basename(dir)
        tarf_name = f"{LCS_PATH}/kepler/{id4}.tar"
        if not os.path.isdir(f"{LCS_PATH}/kepler/{id4}"):
            continue
        files_in = glob.glob(f"{LCS_PATH}/kepler/{id4}/*/*.fits")
        with tarfile.open(tarf_name, mode="a") as tar:
            members = tar.getnames()
            for file in files_in:
                arcname = "/".join(file.split("/")[-3:])
                if version not in os.path.basename(file) or arcname in members:
                    continue
                tar.add(file, arcname=arcname)
                if delete:
                    os.remove(file)

    print("Done!")


def apply_zero_point(dir, quarter):

    # list all fits files in dir
    print(f"Working on {dir:04}")
    files = glob.glob(
        f"{LCS_PATH}/kepler/{dir:04}/*/hlsp_kbonus-bkg_kepler_kepler_*-q{quarter:02}_*_lc.fits"
    )
    print(f"Total files {len(files)}")

    # loaf zero point files
    zp_files = sorted(
        glob.glob(
            f"{PACKAGEDIR}/data/support/zero_points/zero_point_ch*_q{quarter:02}.dat"
        )
    )
    zp = []
    for f in zp_files:
        zp.append(np.loadtxt(f))
    zp = pd.DataFrame(zp, columns=["quarter", "channel", "psf_zp", "sap_zp"])
    zp.quarter = zp.quarter.astype(int)
    zp.channel = zp.channel.astype(int)
    zp.set_index("channel", drop=True, inplace=True)

    for f in tqdm(files, total=len(files), desc="Correcting FITS"):
        try:
            iszp_corr = fits.getval(f, "PSFMZP")
        except KeyError:
            iszp_corr = False
        if iszp_corr:
            continue
        else:
            with fits.open(f, mode="update") as hdul:
                hdul[1].data["FLUX"] *= zp.loc[hdul[0].header["CHANNEL"], "psf_zp"]
                hdul[1].data["PSF_FLUX_NVS"] *= zp.loc[
                    hdul[0].header["CHANNEL"], "psf_zp"
                ]
                hdul[0].header["PSFMZP"] = True
                hdul.flush()


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
        default="fvaT_bkgT_augT_sgmF_iteT_cbvT",
        help="File prefix",
    )
    parser.add_argument(
        "--do-tar",
        dest="do_tarball",
        action="store_true",
        default=False,
        help="Tarball archive",
    )
    parser.add_argument(
        "--delete",
        dest="delete",
        action="store_true",
        default=False,
        help="Delete original fits files after creating tarball",
    )
    args = parser.parse_args()
    if args.quarter is not None and args.channel is not None:
        do_archive(args.quarter, args.channel, suffix=args.suffix)

    if args.dir:
        drop_duplicates(args.dir)
        make_tarball_archive(folders=args.dir, delete=args.delete)
    if args.do_tarball:
        make_tarball_archive(folders=args.dir, delete=args.delete)
