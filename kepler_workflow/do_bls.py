import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
import fitsio
from tqdm import tqdm

from astropy.coordinates import SkyCoord, match_coordinates_3d
import astropy.units as u

from paths import *
from data_quality_assessment_fxs import *

import warnings

warnings.filterwarnings("ignore", category=lk.LightkurveWarning)

dmc_dict = {}


def fancy_flatten(lc, period=None, t0=None, duration=None, plot=False):

    lc = lc.remove_nans()
    lc = lc[lc.flux_err > 0]
    if len(lc) == 0:
        return None
    times = lc.time.bkjd
    key = f"q{lc.quarter:02}_ch{lc.channel:02}"
    if not key in dmc_dict.keys():
        # creating Design matrix
        breaks = list(np.where(np.diff(times) > 0.3)[0] + 1)
        # spline DM
        n_knots = int((times[-1] - times[0]) / 0.5)
        spline_dm = lk.designmatrix.create_spline_matrix(
            times,
            n_knots=n_knots,
            include_intercept=True,
            name="spline_dm",
        )  # .split(breaks)
        spline_dm = lk.DesignMatrix(
            spline_dm.X[:, spline_dm.X.sum(axis=0) != 0], name="spline_dm"
        )
        # offset matrix
        offset_dm = lk.DesignMatrix(
            np.ones_like(times), name="offset"
        )  # .split(breaks)
        offset_dm.prior_mu = np.ones(1)
        offset_dm.prior_sigma = np.ones(1) * 0.000001

        # DM with CBVs, first 4 only
        cbvs = lk.correctors.download_kepler_cbvs(
            mission="Kepler", quarter=lc.quarter, channel=lc.channel
        )
        basis = 4
        cbv_dm = cbvs[
            np.in1d(cbvs.cadenceno.value, lc.cadenceno.value)
        ].to_designmatrix()
        cbv_dm = lk.DesignMatrix(cbv_dm.X[:, :basis], name="cbv").split(breaks)
        cbv_dm.prior_mu = np.ones(basis * (len(breaks) + 1))
        cbv_dm.prior_sigma = np.ones(basis * (len(breaks) + 1)) * 100000

        # collection of DMs
        dmc = lk.DesignMatrixCollection([offset_dm, cbv_dm, spline_dm])
        dmc_dict[key] = dmc
    else:
        dmc = dmc_dict[key]

    cadence_mask = np.ones(len(times), dtype=bool)
    if period and t0 and duration:
        n = 0
        while t0 + n * period <= times.max():
            if t0 + n * period < times.min():
                n += 1
                continue
            idx = np.where(
                (times >= t0 + n * period - duration / 2)
                & (times <= t0 + n * period + duration / 2)
            )[0]
            cadence_mask[idx] = False
            n += 1
    # print(f"Masking total cadences {(~cadence_mask).sum()}")
    rc = lk.RegressionCorrector(lc)
    clc = rc.correct(dmc, sigma=3, cadence_mask=cadence_mask)
    if plot:
        # fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        # spline_dm.plot(ax=ax[0])
        # offset_dm.plot(ax=ax[1])
        # cbv_dm.plot(ax=ax[2])
        # plt.show()
        rc.diagnose()
        plt.show()

    return clc


def do_bls(quarter, dir):
    kbonus = pd.read_csv(
        f"{PACKAGEDIR}/data/catalogs/tpf/kbonus_catalog_q05.csv", index_col=0
    )

    koi = pd.read_csv(
        f"{ARCHIVE_PATH}/data/catalogs/NASA_exoplanet_archive_2021.11.10_15.59.09.csv",
        comment="#",
    )
    koi_fp = koi.query("koi_disposition == 'FALSE POSITIVE' and koi_fpflag_co == 1")

    lcfs = sorted(
        glob.glob(
            f"{LCS_PATH}/kepler/{dir}/*/"
            f"hlsp_kbonus-kbkgd_kepler_kepler_*-q{quarter:02}_kepler_v1.0_lc.fits"
        )
    )
    print(f"Total LCFs in {dir}: {len(lcfs)}")

    # BLS code
    sap_clcs, psf_clcs = [], []
    bls_res_psf, bls_res_sap = [], []
    power_psf, power_sap = [], []
    periods = []
    gaiaid, kepid = [], []

    pmin, pmax, ffrac = 0.35, 30, 30
    dur = np.linspace(0.01, 0.34, 5)

    for k, f in tqdm(enumerate(lcfs), total=len(lcfs)):
        lc = lk.KeplerLightCurve.read(f)
        kic = int(lc.KEPLERID) if lc.KEPLERID != "" else 0

        # check if sources is a KOI FP
        if kic in koi_fp.kepid.values:
            idx = np.where(kic == koi_fp.kepid.values)[0]
            p0 = koi_fp.iloc[idx]["koi_period"].values[0]
            t0 = koi_fp.iloc[idx]["koi_time0bk"].values[0]
            d0 = koi_fp.iloc[idx]["koi_duration"].values[0] / 24.0
            if p0 < 40:
                pmin = p0 * 0.9
                pmax = p0 * 1.1
        else:
            p0 = None
            t0 = None
            d0 = None

        # BLS for PSF
        if np.isfinite(lc.flux).all():
            nclc_psf = fancy_flatten(lc, period=p0, t0=t0, duration=d0, plot=False)
            if isinstance(nclc_psf, lk.KeplerLightCurve):
                nclc_psf = nclc_psf.normalize()
                psf_clcs.append(nclc_psf)
                psf_per = nclc_psf.to_periodogram(
                    method="bls",
                    minimum_period=pmin,
                    maximum_period=pmax,
                    frequency_factor=ffrac,
                    duration=dur,
                )
                power_psf.append(psf_per.power)
                periods.append(psf_per.period)
                try:
                    stats = psf_per.compute_stats(
                        period=psf_per.period_at_max_power,
                        duration=psf_per.duration_at_max_power,
                        transit_time=psf_per.transit_time_at_max_power,
                    )
                    depth_odd = stats["depth_odd"][0].value
                    depth_odd_e = stats["depth_odd"][1].value
                    depth_even = stats["depth_even"][0].value
                    depth_even_e = stats["depth_even"][1].value
                except:
                    depth_odd = np.nan
                    depth_odd_e = np.nan
                    depth_even = np.nan
                    depth_even_e = np.nan

                bls_res_psf.append(
                    [
                        lc.GAIAID,
                        f"KIC {kic:09}" if kic != 0 else None,
                        psf_per.period_at_max_power.value,
                        psf_per.transit_time_at_max_power.value,
                        psf_per.duration_at_max_power.value,
                        psf_per.depth_at_max_power.value,
                        depth_odd,
                        depth_odd_e,
                        depth_even,
                        depth_even_e,
                        psf_per.snr[np.argmax(psf_per.power)].value,
                    ]
                )

        # BLS for SAP
        if np.isfinite(lc.sap_flux).all():
            lc.flux = lc.sap_flux
            lc.flux_err = lc.sap_flux_err
            nclc_sap = fancy_flatten(lc, period=p0, t0=t0, duration=d0)
            if isinstance(nclc_sap, lk.KeplerLightCurve):
                nclc_sap = nclc_sap.normalize()
                sap_clcs.append(nclc_sap)

                sap_per = nclc_sap.to_periodogram(
                    method="bls",
                    minimum_period=pmin,
                    maximum_period=pmax,
                    frequency_factor=ffrac,
                    duration=dur,
                )
                power_sap.append(sap_per.power)
                try:
                    stats = sap_per.compute_stats(
                        period=sap_per.period_at_max_power,
                        duration=sap_per.duration_at_max_power,
                        transit_time=sap_per.transit_time_at_max_power,
                    )
                    depth_odd = stats["depth_odd"][0].value
                    depth_odd_e = stats["depth_odd"][1].value
                    depth_even = stats["depth_even"][0].value
                    depth_even_e = stats["depth_even"][1].value
                except:
                    depth_odd = np.nan
                    depth_odd_e = np.nan
                    depth_even = np.nan
                    depth_even_e = np.nan
                bls_res_sap.append(
                    [
                        lc.GAIAID,
                        f"KIC {kic:09}" if kic != 0 else None,
                        sap_per.period_at_max_power.value,
                        sap_per.transit_time_at_max_power.value,
                        sap_per.duration_at_max_power.value,
                        sap_per.depth_at_max_power.value,
                        depth_odd,
                        depth_odd_e,
                        depth_even,
                        depth_even_e,
                        sap_per.snr[np.argmax(sap_per.power)].value,
                    ]
                )

    cols = [
        "gaia_id",
        "kepid",
        "period",
        "t0",
        "duration",
        "depth",
        "depth_odd",
        "depth_odd_e",
        "depth_even",
        "depth_even_e",
        "snr",
    ]
    bls_res_psf = pd.DataFrame(bls_res_psf, columns=cols)
    bls_res_sap = pd.DataFrame(bls_res_sap, columns=cols)

    bls_res_psf.to_csv(
        f"{PACKAGEDIR}/data/bls/q{quarter:02}/bls_results_q{quarter:02}_{dir}_psf.csv"
    )
    bls_res_sap.to_csv(
        f"{PACKAGEDIR}/data/bls/q{quarter:02}/bls_results_q{quarter:02}_{dir}_sap.csv"
    )

    # to_save = {
    #     "periods": periods[0],
    #     "power_psf": np.array(power_psf),
    #     "power_sap": np.array(power_sap),
    # }
    np.savez(
        f"{PACKAGEDIR}/data/bls/q{quarter:02}/bls_power_q{quarter:02}_{dir}.npz",
        periods=periods[0],
        power_psf=np.asarray(power_psf),
        power_sap=np.asarray(power_sap),
    )

    return


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
    args = parser.parse_args()

    do_bls(args.quarter, args.dir)
