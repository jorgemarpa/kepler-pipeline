import os, sys
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from glob import glob
import fitsio

sys.path.append(f"{os.path.dirname(os.getcwd())}/kepler_workflow/")
from paths import ARCHIVE_PATH, OUTPUT_PATH, LCS_PATH, PACKAGEDIR


def main(quarter=1, targets="mstars"):
    fnames = sorted(
        glob(
            f"{LCS_PATH}/kepler/{targets}/*/*/"
            f"hlsp_kbonus-bkg_kepler_kepler_*-q{quarter:02}_kepler_v1.1.1_lc.fits"
        )
    )
    print(f"Files in {len(fnames)}")
    psf, psf_e, psf_nova, psf_nova_e, sap, sap_e, chi2 = [], [], [], [], [], [], []
    gaia_desig, kic, ra, dec = [], [], [], []
    time, cadn = [], []
    for f in tqdm(fnames, leave=True):
        fits = fitsio.FITS(f)
        gaia_desig.append(fits[0].read_header()["GAIAID"])
        kic.append(fits[0].read_header()["KEPLERID"])
        ra.append(fits[0].read_header()["RA_OBJ"])
        dec.append(fits[0].read_header()["DEC_OBJ"])

        time.append(fits[1]["TIME"].read())
        cadn.append(fits[1]["CADENCENO"].read())

        psf.append(fits[1]["FLUX"].read())
        psf_e.append(fits[1]["FLUX_ERR"].read())
        psf_nova.append(fits[1]["PSF_FLUX_NOVA"].read())
        psf_nova_e.append(fits[1]["PSF_FLUX_ERR_NOVA"].read())
        sap.append(fits[1]["SAP_FLUX"].read())
        sap_e.append(fits[1]["SAP_FLUX_ERR"].read())
        chi2.append(fits[1]["RED_CHI2"].read())

    print("Saving coord")
    df = pd.DataFrame(
        [gaia_desig, kic, ra, dec], index=["gaia_designation", "kic", "ra", "dec"]
    ).T.set_index("gaia_designation")
    df.replace("", np.nan).reset_index().to_csv(
        f"{LCS_PATH}/kepler/{targets}/feather/{targets}_q{quarter:02}_coord.feather"
    )
    print("Saving times")
    pd.DataFrame(
        [np.array(cadn[0], dtype=int), np.array(time[0], dtype=float)],
        index=["cadenceno", "bkjd"],
    ).T.to_feather(
        f"{LCS_PATH}/kepler/{targets}/feather/{targets}_q{quarter:02}_time.feather"
    )
    print("Saving photometry")
    pd.DataFrame(psf, index=gaia_desig).T.to_feather(
        f"{LCS_PATH}/kepler/{targets}/feather/{targets}_q{quarter:02}_psf.feather"
    )
    pd.DataFrame(psf_e, index=gaia_desig).T.to_feather(
        f"{LCS_PATH}/kepler/{targets}/feather/{targets}_q{quarter:02}_psf_e.feather"
    )
    pd.DataFrame(psf_nova, index=gaia_desig).T.to_feather(
        f"{LCS_PATH}/kepler/{targets}/feather/{targets}_q{quarter:02}_psf_nova.feather"
    )
    pd.DataFrame(psf_nova_e, index=gaia_desig).T.to_feather(
        f"{LCS_PATH}/kepler/{targets}/feather/{targets}_q{quarter:02}_psf_nova_e.feather"
    )
    pd.DataFrame(sap, index=gaia_desig).T.to_feather(
        f"{LCS_PATH}/kepler/{targets}/feather/{targets}_q{quarter:02}_sap.feather"
    )
    pd.DataFrame(sap_e, index=gaia_desig).T.to_feather(
        f"{LCS_PATH}/kepler/{targets}/feather/{targets}_q{quarter:02}_sap_e.feather"
    )
    pd.DataFrame(chi2, index=gaia_desig).T.to_feather(
        f"{LCS_PATH}/kepler/{targets}/feather/{targets}_q{quarter:02}_chi2.feather"
    )
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoEncoder")
    parser.add_argument(
        "--quarter",
        dest="quarter",
        type=int,
        default=None,
        help="Quarter number.",
    )
    args = parser.parse_args()
    main(quarter=args.quarter)
