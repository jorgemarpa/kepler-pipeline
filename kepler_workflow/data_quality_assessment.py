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

from paths import PACKAGEDIR, ARCHIVE_PATH, LCS_PATH
from data_quality_assessment_fxs import *

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def drop_repeated(lcs):
    """
    Return index of duplicated light curves, we keep the good ones.
    """
    gids = np.array([int(x.GAIAID.split(" ")[-1]) for x in lcs])
    dup_mask = np.ones_like(gids, dtype=bool)
    dup_mask[np.unique(gids, return_index=True)[1]] = False

    drop_idx = []
    for dup in gids[dup_mask]:
        dup_idx = np.where(dup == gids)[0]
        err_means = np.array([lcs[idx].flux_err.mean().value for idx in dup_idx])
        drop_idx.extend(dup_idx[np.isnan(err_means)])
        dup_idx = dup_idx[np.isfinite(err_means)]
        err_means = err_means[np.isfinite(err_means)]
        drop_idx.extend(np.delete(dup_idx, np.argmin(err_means)))
    return drop_idx


def main(quarter, channel, download=False):
    tar_file = np.sort(
        glob.glob(
            f"{LCS_PATH}/kepler/ch{channel:02}/q{quarter:02}/"
            f"kbonus-bkgd_ch{channel:02}_q{quarter:02}_v1.0_lc*_"
            f"poscorT_sqrt_tk6_tp100.tar.gz"
        )
    )
    if len(tar_file) == 0:
        print("No light curve archive...")
        sys.exit()
    print(tar_file)
    if not os.path.isfile(tar_file[0]):
        print("No light curve archive...")
        sys.exit()

    lcs, kics, tpfs_org = get_archive_lightcurves(tar_file)
    print(len(lcs), len(kics), len(tpfs_org))
    drop_idx = drop_repeated(lcs)
    print(drop_idx)
    for k in sorted(drop_idx)[::-1]:
        del lcs[k], kics[k], tpfs_org[k]
    print(len(lcs), len(kics), len(tpfs_org))
    jm_stats = compute_stats_from_lcs(lcs, project="kbonus", do_cdpp=True)

    kplcs, kplcs_exist = get_keple_lightcurves(
        kics, quarter, tar=False if quarter == 5 else True
    )
    print(kplcs_exist)
    # kplcs_exist = ~np.all([lc == None for lc in kplcs])
    if (not kplcs_exist) and (download):
        print("Downloading Kepler LCFs")
        pr_name = make_lc_download_sh(list(set(kics)), channel, quarter)
        # Use subprocess to run the shell script and download the LCFs
        # This takes time to run
        exit = subprocess.run(["sh", "pr_name"], stdout=subprocess.DEVNULL)
        print("The exit code was: %d" % exit.returncode)

    if kplcs_exist:
        kp_stats = compute_stats_from_lcs(kplcs, project="kepler", do_cdpp=True)
        psf_zp, _, _, _ = compute_zero_point(
            jm_stats["lc_mean_psf"], kp_stats["lc_mean_pdc"], use_ransac=False
        )
        sap_zp, _, _, _ = compute_zero_point(
            jm_stats["lc_mean_sap"], kp_stats["lc_mean_pdc"], use_ransac=False
        )
        zp_fname = (
            f"{PACKAGEDIR}/data/support/zero_point_ch{channel:02}_q{quarter:02}.dat"
        )
        np.savetxt(
            zp_fname,
            np.array([[quarter, channel, psf_zp, sap_zp]]),
            header="quarter,channel,psf_zp,sap_zp",
            delimiter=",",
            fmt="%i %i %f %f",
        )
        # sys.exit()
        feat_kp_sap = get_features(kplcs, flux_col="sap_flux")
        feat_kp_pdc = get_features(kplcs, flux_col="pdcsap_flux")
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
