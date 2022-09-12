import os
import sys
import glob
import argparse
import tarfile
import tempfile
import numpy as np
import pandas as pd
import lightkurve as lk
import fitsio
from tqdm import tqdm

from paths import *


def main(dir, quarter, version="1.1.1", archive_tar=False):
    print(f"Working on {dir}")
    if archive_tar:
        print("Archive is tarball")
        tarname = f"{LCS_PATH}/kepler/{dir}.tar"
        if not os.path.isfile(tarname):
            print("Tar file does not exist")
            sys.exit()
        tar = tarfile.open(tarname, mode="r")
        lcfs = tar.getnames()
        lcfs = [x for x in lcfs if f"q{quarter:02}" in x]
        tmpdir = tempfile.TemporaryDirectory(prefix="temp_fits")
    else:
        lcfs = glob.glob(
            f"{LCS_PATH}/kepler/{dir}/*/"
            f"hlsp_kbonus-bkg_kepler_kepler_*-q{quarter:02}_kepler_v{version}_lc.fits"
        )
    print(f"Total files {len(lcfs)}")
    kics, gids = [], []
    ras, decs = [], []
    column, row = [], []
    sap_flux, sap_flux_err = [], []
    psf_flux, psf_flux_err = [], []
    FLFRCSAP, CROWDSAP, NPIXSAP, PSFFRAC, PERTRATI, PERTSTD = [], [], [], [], [], []
    channel = []
    gmag, rpmag, bpmag = [], [], []

    for k, f in tqdm(enumerate(lcfs), total=len(lcfs), desc="FITS"):
        if archive_tar:
            tar.extract(f, tmpdir.name)
            f = f"{tmpdir.name}/{f}"

        gids.append(fitsio.read_header(f)["GAIAID"])
        kics.append(fitsio.read_header(f)["KEPLERID"])
        ras.append(fitsio.read_header(f)["RA_OBJ"])
        decs.append(fitsio.read_header(f)["DEC_OBJ"])
        column.append(fitsio.read_header(f)["COLUMN"])
        row.append(fitsio.read_header(f)["ROW"])
        FLFRCSAP.append(fitsio.read_header(f)["FLFRCSAP"])
        CROWDSAP.append(fitsio.read_header(f)["CROWDSAP"])
        NPIXSAP.append(fitsio.read_header(f)["NPIXSAP"])
        PSFFRAC.append(fitsio.read_header(f)["PSFFRAC"])
        PERTRATI.append(fitsio.read_header(f)["PERTRATI"])
        PERTSTD.append(fitsio.read_header(f)["PERTSTD"])
        channel.append(fitsio.read_header(f)["CHANNEL"])
        gmag.append(fitsio.read_header(f)["GMAG"])
        rpmag.append(fitsio.read_header(f)["RPMAG"])
        bpmag.append(fitsio.read_header(f)["BPMAG"])

        sap_flux.append(np.nanmedian(fitsio.read(f, ext=1, columns="SAP_FLUX")))
        psf_flux.append(np.nanmedian(fitsio.read(f, ext=1, columns="FLUX")))

        nt = fitsio.read_header(f, ext=1)["NAXIS2"]
        sap_flux_err.append(
            np.sqrt(np.nansum(fitsio.read(f, ext=1, columns="SAP_FLUX_ERR") ** 2)) / nt
        )
        psf_flux_err.append(
            np.sqrt(np.nansum(fitsio.read(f, ext=1, columns="FLUX_ERR") ** 2)) / nt
        )
        os.remove(f)
    if archive_tar:
        tar.close()
        tmpdir.cleanup()

    df = pd.DataFrame.from_dict(
        {
            "kic": kics,
            "gaia_designation": gids,
            "ra": ras,
            "dec": decs,
            "column": column,
            "row": row,
            "sap_flux": sap_flux,
            "sap_flux_err": sap_flux_err,
            "psf_flux": psf_flux,
            "psf_flux_err": psf_flux_err,
            "gmag": gmag,
            "rpmag": rpmag,
            "bpmag": bpmag,
            "channel": channel,
            "flfrcsap": FLFRCSAP,
            "crowdsap": CROWDSAP,
            "npixsap": NPIXSAP,
            "psffrac": PSFFRAC,
            "pertrati": PERTRATI,
            "pertstd": PERTSTD,
        }
    )
    dirname = f"{KBONUS_CAT_PATH}/tpf/"
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    df.to_csv(f"{KBONUS_CAT_PATH}/tpf/kbonus_catalog_q{quarter:02}_dir{dir}.csv")


def concat_dir_catalogs(quarter):
    files = glob.glob(f"{KBONUS_CAT_PATH}/tpf/kbonus_catalog_q{quarter:02}_dir*.csv")

    df = pd.concat([pd.read_csv(x) for x in files])
    for x in files:
        os.remove(x)

    df.reset_index(drop=True).drop("Unnamed: 0", axis=1).to_csv(
        f"{KBONUS_CAT_PATH}/tpf/kbonus_catalog_q{quarter:02}.csv"
    )


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
        "--dir",
        dest="dir",
        type=str,
        default=None,
        help="Kepler 4-digit directory",
    )
    parser.add_argument(
        "--concat",
        dest="concat",
        action="store_true",
        default=False,
        help="Concatenate dir catalogs.",
    )
    parser.add_argument(
        "--tar",
        dest="tar",
        action="store_true",
        default=False,
        help="Tarball archive",
    )
    args = parser.parse_args()

    if args.concat:
        concat_dir_catalogs(args.quarter)
    else:
        main(args.dir, args.quarter, archive_tar=args.tar)
