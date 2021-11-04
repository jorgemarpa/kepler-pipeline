import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import lightkurve as lk

import os, sys

path = os.path.dirname(os.getcwd())

sys.path.append(f"{path}/kepler_workflow/")

from data_quality_assessment_fxs import *

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main(quarter, channel):
    tar_file = (
        f"../data/lcs/kepler/ch{channel:02}/q{quarter:02}/"
        f"kbonus-bkgd_ch{channel:02}_q{quarter:02}_v1.0_lc_poscorT_sqrt_tk6_tp100.tar.gz"
    )
    print(tar_file)
    if not os.path.isfile(tar_file):
        print("No light curve archive...")
        sys.exit()

    lcs, kics, tpfs_org = get_archive_lightcurves(tar_file)
    jm_stats = compute_stats_from_lcs(lcs, project="kbonus")

    kplcs = get_keple_lightcurves(kics, quarter)
    kplcs_exist = ~np.all([lc == None for lc in kplcs])
    print(kplcs_exist)

    if kplcs_exist:
        feat_kp_sap = get_features(kplcs, flux_col="sap_flux")
        feat_kp_pdc = get_features(kplcs, flux_col="pdcsap_flux")
        kp_stats = compute_stats_from_lcs(kplcs, project="kepler")
        psf_zp, _, _, _ = compute_zero_point(
            jm_stats["lc_mean_psf"], kp_stats["lc_mean_pdc"], use_ransac=False
        )
        sap_zp, _, _, _ = compute_zero_point(
            jm_stats["lc_mean_sap"], kp_stats["lc_mean_pdc"], use_ransac=False
        )
    else:
        kplcs = None
        feat_kp_sap = None
        feat_kp_pdc = None
        kp_stats = None
        psf_zp = 0.8
        sap_zp = 1.09

    jm_stats["lc_mean_psf_zp"] = jm_stats["lc_mean_psf"] * psf_zp

    meta = {
        "channel": channel,
        "quarter": quarter,
        "psf_zp": psf_zp,
        "sap_zp": sap_zp,
    }

    feat_jm_sap = get_features(lcs, flux_col="sap_flux")
    feat_jm_psf = get_features(lcs, flux_col="flux")
    feat_jm_psfnv = get_features(lcs, flux_col="psf_flux_nvs")

    features = {
        "feat_jm_sap": feat_jm_sap,
        "feat_jm_psf": feat_jm_psf,
        "feat_jm_psfnv": feat_jm_psfnv,
        "feat_kp_sap": feat_kp_sap,
        "feat_kp_pdc": feat_kp_pdc,
    }

    lightcurves = {"lcs": lcs, "kplcs": kplcs}
    stats = {
        "kp_stats": kp_stats,
        "jm_stats": jm_stats,
    }
    meta = {
        "channel": channel,
        "quarter": quarter,
        "psf_zp": psf_zp,
        "sap_zp": sap_zp,
    }

    print("Creating Dashboard")
    make_dashboard(stats, features, lightcurves, meta, save=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQA")
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
    args = parser.parse_args()

    main(args.quarter, args.channel)
