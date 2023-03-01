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
from scipy import stats

from paths import *

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main(dir, quarter, version="1.0", archive_tar=False, tar_gz=False):
    print(f"Working on {dir}")
    if archive_tar:
        print("Archive is tarball")
        tarname = f"{LCS_PATH}/kepler/{dir}.tar{'.gz' if tar_gz else ''}"
        if not os.path.isfile(tarname):
            print("Tar file does not exist")
            sys.exit()
        tar = tarfile.open(tarname, mode="r:gz" if tar_gz else "r")
        lcfs = tar.getnames()
        if "_s" not in dir and quarter != "all":
            lcfs = [x for x in lcfs if f"q{int(quarter):02}" in x]
        else:
            lcfs = [x for x in lcfs if ".fits" in x]
        tmpdir = tempfile.TemporaryDirectory(prefix="temp_fits")
    else:
        if quarter == "all":
            lcfs = glob.glob(
                f"{LCS_PATH}/kepler/{dir}/*/"
                f"hlsp_kbonus-bkg_kepler_kepler_*_kepler_v{version}_lc.fits"
            )
        else:
            lcfs = glob.glob(
                f"{LCS_PATH}/kepler/{dir}/*/"
                f"hlsp_kbonus-bkg_kepler_kepler_*-q{int(quarter):02}_kepler_v{version}_lc.fits"
            )
    print(f"Total files {len(lcfs)}")
    kics, gids, fname = [], [], []
    ras, decs = [], []
    TPFORG = []
    SAP_FLUX, SAP_FLUX_ERR = [], []
    PSF_FLUX, PSF_FLUX_ERR = [], []
    FLFRCSAP, CROWDSAP, NPIXSAP, PSFFRAC, PERTRATI, PERTSTD = [], [], [], [], [], []
    PSF_AVAIL, SAP_AVAIL, QDETECT = [], [], []
    PSF_CDPP, SAP_CDPP, SAP_PPP, PSF_PPP = [], [], [], []

    for k, f in tqdm(enumerate(lcfs), total=len(lcfs), desc="FITS"):
        if archive_tar:
            tar.extract(f, tmpdir.name)
            f = f"{tmpdir.name}/{f}"

        fname.append(f.split("_")[-4])
        gids.append(fitsio.read_header(f, ext=0)["GAIAID"])
        kics.append(fitsio.read_header(f, ext=0)["KEPLERID"])
        ras.append(fitsio.read_header(f, ext=0)["RA_OBJ"])
        decs.append(fitsio.read_header(f, ext=0)["DEC_OBJ"])
        QDETECT.append(fitsio.read_header(f, ext=0)["QDETECT"])
        TPFORG.append(fitsio.read_header(f, ext=0)["TPFORG"])

        psf_flux = fitsio.read(f, ext=1, columns="FLUX")
        sap_flux = fitsio.read(f, ext=1, columns="SAP_FLUX")
        SAP_FLUX.append(np.nanmedian(sap_flux))
        PSF_FLUX.append(np.nanmedian(psf_flux))

        psf_flux_err = fitsio.read(f, ext=1, columns="FLUX_ERR")
        sap_flux_err = fitsio.read(f, ext=1, columns="SAP_FLUX_ERR")
        nt = fitsio.read_header(f, ext=1)["NAXIS2"]
        SAP_FLUX_ERR.append(
            np.sqrt(np.nansum(fitsio.read(f, ext=1, columns="SAP_FLUX_ERR") ** 2)) / nt
        )
        PSF_FLUX_ERR.append(
            np.sqrt(np.nansum(fitsio.read(f, ext=1, columns="FLUX_ERR") ** 2)) / nt
        )

        time = fitsio.read(f, ext=1, columns="TIME")
        if np.isfinite(psf_flux).all() & (psf_flux != 0).all():
            PSF_PPP.append(
                (1.48 / np.sqrt(2))
                * stats.median_abs_deviation(psf_flux[1:] - psf_flux[:-1])
            )
            lc = lk.LightCurve(time=time, flux=psf_flux, flux_err=psf_flux_err)
            PSF_CDPP.append(lc.estimate_cdpp().value)
        else:
            PSF_PPP.append(np.nan)
            PSF_CDPP.append(np.nan)
        if np.isfinite(sap_flux).all() & (sap_flux != 0).all():
            SAP_PPP.append(
                (1.48 / np.sqrt(2))
                * stats.median_abs_deviation(sap_flux[1:] - sap_flux[:-1])
            )
            lc = lk.LightCurve(time=time, flux=sap_flux, flux_err=sap_flux_err)
            SAP_CDPP.append(lc.estimate_cdpp().value)
        else:
            SAP_PPP.append(np.nan)
            SAP_CDPP.append(np.nan)

        psf_av = [0] * 18
        sap_av = [0] * 18
        if "_s" in dir:
            hdul = fitsio.FITS(f)
            lc_ext = [
                k for k in range(len(hdul)) if "LIGHTCURVE_Q" in hdul[k].get_extname()
            ]
            FLFRCSAP_aux = []
            CROWDSAP_aux = []
            NPIXSAP_aux = []
            PSFFRAC_aux = []
            PERTRATI_aux = []
            PERTSTD_aux = []
            psf_av = [0] * 18
            sap_av = [0] * 18
            for ext in lc_ext:
                FLFRCSAP_aux.append(fitsio.read_header(f, ext=ext)["FLFRCSAP"])
                CROWDSAP_aux.append(fitsio.read_header(f, ext=ext)["CROWDSAP"])
                NPIXSAP_aux.append(fitsio.read_header(f, ext=ext)["NPIXSAP"])
                PSFFRAC_aux.append(fitsio.read_header(f, ext=ext)["PSFFRAC"])
                PERTRATI_aux.append(fitsio.read_header(f, ext=ext)["PERTRATI"])
                PERTSTD_aux.append(fitsio.read_header(f, ext=ext)["PERTSTD"])
                psf_flux_q = fitsio.read(f, ext=ext, columns="FLUX")
                sap_flux_q = fitsio.read(f, ext=ext, columns="SAP_FLUX")
                psf_av[fitsio.read_header(f, ext=ext)["QUARTER"]] = (
                    1
                    if (np.isfinite(psf_flux_q).all() & (psf_flux_q != 0).all())
                    else 0
                )
                sap_av[fitsio.read_header(f, ext=ext)["QUARTER"]] = (
                    1
                    if (np.isfinite(sap_flux_q).all() & (sap_flux_q != 0).all())
                    else 0
                )
            FLFRCSAP.append(np.nanmin(FLFRCSAP_aux))
            CROWDSAP.append(np.nanmin(CROWDSAP_aux))
            NPIXSAP.append(np.nanmin(NPIXSAP_aux))
            PSFFRAC.append(np.nanmin(PSFFRAC_aux))
            PERTRATI.append(np.nanmedian(PERTRATI_aux))
            PERTSTD.append(np.nanmin(PERTSTD_aux))
            PSF_AVAIL.append(''.join([str(x) for x in psf_av]))
            SAP_AVAIL.append(''.join([str(x) for x in sap_av]))

        else:
            FLFRCSAP.append(fitsio.read_header(f)["FLFRCSAP"])
            CROWDSAP.append(fitsio.read_header(f)["CROWDSAP"])
            NPIXSAP.append(fitsio.read_header(f)["NPIXSAP"])
            PSFFRAC.append(fitsio.read_header(f)["PSFFRAC"])
            PERTRATI.append(fitsio.read_header(f)["PERTRATI"])
            PERTSTD.append(fitsio.read_header(f)["PERTSTD"])
            PSF_AVAIL.append(
                1 if (np.isfinite(psf_flux_q).all() & (psf_flux_q != 0).all()) else 0
            )
            SAP_AVAIL.append(
                1 if (np.isfinite(sap_flux_q).all() & (sap_flux_q != 0).all()) else 0
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
            "sap_flux": SAP_FLUX,
            "sap_flux_err": SAP_FLUX_ERR,
            "psf_flux": PSF_FLUX,
            "psf_flux_err": PSF_FLUX_ERR,
            "flfrcsap": FLFRCSAP,
            "crowdsap": CROWDSAP,
            "npixsap": NPIXSAP,
            "psffrac": PSFFRAC,
            "pertrati": PERTRATI,
            "pertstd": PERTSTD,
            "psf_avail": PSF_AVAIL,
            "sap_avail": SAP_AVAIL,
            "fname": fname,
            "qdetect": QDETECT,
            "tpforg": TPFORG,
            "psf_cdpp": PSF_CDPP,
            "sap_cdpp": SAP_CDPP,
            "sap_ppp": SAP_PPP,
            "psf_ppp": PSF_PPP,
        }
    )
    dirname = f"{KBONUS_CAT_PATH}/tpf/"
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    if quarter == "all":
        df.to_csv(f"{KBONUS_CAT_PATH}/tpf/kbonus_catalog_dir{dir}.csv")
    else:
        df.to_csv(
            f"{KBONUS_CAT_PATH}/tpf/kbonus_catalog_q{int(quarter):02}_dir{dir}.csv"
        )


def concat_dir_catalogs(quarter):
    files = glob.glob(
        f"{KBONUS_CAT_PATH}/tpf/kbonus_catalog_q{int(quarter):02}_dir*.csv"
    )

    df = pd.concat([pd.read_csv(x) for x in files])
    for x in files:
        os.remove(x)

    df.reset_index(drop=True).drop("Unnamed: 0", axis=1).to_csv(
        f"{KBONUS_CAT_PATH}/tpf/kbonus_catalog_q{int(quarter):02}.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quarter",
        dest="quarter",
        type=str,
        default="all",
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
    parser.add_argument(
        "--gz",
        dest="gz",
        action="store_true",
        default=False,
        help="Gzip file",
    )
    args = parser.parse_args()

    if args.concat:
        concat_dir_catalogs(args.quarter)
    else:
        main(args.dir, args.quarter, archive_tar=args.tar, tar_gz=args.gz)
