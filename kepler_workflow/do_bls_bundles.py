import os, sys
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from glob import glob
import lightkurve as lk
from astropy.io import fits
import astropy.units as u

sys.path.append(f"{os.path.dirname(os.getcwd())}/kepler_workflow/")
from paths import ARCHIVE_PATH, OUTPUT_PATH, LCS_PATH, PACKAGEDIR
from do_bundles import get_lcs_from_archive, do_bundle

warnings.filterwarnings("ignore", category=lk.LightkurveWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

dmc_dict = {}


def fancy_flatten(
    lc, period=None, t0=None, duration=None, plot=False, correction="sub"
):

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
    if correction == "div":
        clc = lc.copy()
        clc.flux = lc.flux / (np.nanmedian(lc.flux) + rc.model_lc.flux)
        clc.flux_err = lc.flux * np.sqrt(
            (lc.flux_err / lc.flux) ** 2
            + (rc.model_lc.flux_err / rc.model_lc.flux) ** 2
        )
        clc.flux._set_unit(u.electron / u.second)
        clc.flux_err._set_unit(u.electron / u.second)
    if plot:
        # fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        # spline_dm.plot(ax=ax[0])
        # offset_dm.plot(ax=ax[1])
        # cbv_dm.plot(ax=ax[2])
        # plt.show()
        rc.diagnose()
        plt.show()

    return clc


def get_lc(
    name,
    bundle="mstars",
    quarter="all",
    force_phot_column=True,
    correction="div",
    drop=[],
):
    if quarter == "all":
        lc_files = sorted(
            glob(
                f"{LCS_PATH}/kepler/{bundle}/{name[:4]}/{name}/"
                f"hlsp_kbonus-bkg_kepler_kepler_*-{name}-q*_lc.fits"
            )
        )
    else:
        lc_files = sorted(
            glob(
                f"{LCS_PATH}/kepler/{bundle}/{name[:4]}/{name}/"
                f"hlsp_kbonus-bkg_kepler_kepler_*-{name}-q{quarter:02}*_lc.fits"
            )
        )
    if len(lc_files) == 0:
        return None, None, None
    lc = lk.LightCurveCollection([lk.KeplerLightCurve.read(x) for x in lc_files])
    if len(drop) > 0:
        lc = lk.LightCurveCollection([x for x in lc if x.quarter not in drop])
    _phot = "psf"

    if force_phot_column:
        if isinstance(quarter, int):
            if np.isfinite(lc[0].flux).all():
                _phot = "psf"
            elif np.isfinite(lc[0].psf_flux_nova).all():
                _phot = "psf_nova"
                lc[0].flux = lc[0].psf_flux_nova
                lc[0].flux_err = lc[0].psf_flux_err_nova
            elif np.isfinite(lc[0].sap_flux).all():
                _phot = "sap"
                lc[0].flux = lc[0].sap_flux
                lc[0].flux_err = lc[0].sap_flux_err
            else:
                return None, None, None
        else:
            pass

    lc = lk.LightCurveCollection([x for x in lc if np.isfinite(x.flux).all()])
    if len(lc) == 0:
        return None, None, None

    lc_flat = []
    for x in lc:
        try:
            aux = fancy_flatten(x, correction=correction).normalize()
        except:
            aux = x.flatten()
        lc_flat.append(aux)
    lc_flat = lk.LightCurveCollection(lc_flat)

    return lc.stitch(), lc_flat.stitch(), _phot


def do_bls_quarter(
    bundle="mstars",
    quarter=5,
    clean_catalog=True,
    period_range="short",
    force_phot_column=False,
    batch_size=1000,
    batch=-1,
):
    fname = (
        f"{PACKAGEDIR}/data/catalogs/tpf/"
        f"kbonus-bkg_kepler_v1.1.1_source_catalog_{bundle}_allcols.csv"
    )

    df = pd.read_csv(fname)

    if clean_catalog:
        df = df.query("kct_avail_flag < 2")
        df = df[np.isfinite(df.Teff)]

    if batch > 0:
        print("Running in batch mode")
        print(f"Batch N = {batch} Size = {batch_size}")
        start, end = batch_size * (batch - 1), batch_size * (batch)
        if start > len(df):
            raise ValueError("Batch out of range")
        print(f"[{start}, {end}]")
        df = df.iloc[start:end]

    print(f"Catalog Q {quarter} size: {len(df)}")

    names = []
    for k, row in df.iterrows():
        if row.kic < 1.000000e20:
            names.append(int(row.kic))
        else:
            names.append(int(row.gaia_designation.split(" ")[-1]))
    names = [f"{x:09}" if x < 22934493 else str(x) for x in names]

    if period_range == "short":
        pmin, pmax, ffrac = 0.36, 20, 10
    elif period_range == "long":
        pmin, pmax, ffrac = 20, 100, 10
    else:
        pmin, pmax, ffrac = 1, 50, 10

    if quarter == "all":
        quarter_str = quarter
        duration_search = [0.05, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.3, 0.35]
        period_search = 1 / np.linspace(1 / pmin, 1 / pmax, 10000)
    else:
        quarter_str = f"{quarter:02}"
        duration_search = [0.03, 0.05, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.3, 0.34]
        period_search = 1 / np.linspace(1 / pmin, 1 / pmax, 75000)

    print(
        f"Doing {period_range} Period range "
        f"[{period_search.min():02}, {period_search.max():02}] "
        f"{len(period_search)} points"
    )
    gaia_designation, phot = [], []
    best_periods, periods_snr, best_depth, best_depth_e, best_duration, tt0 = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    power = []

    for k, nam in tqdm(enumerate(names), total=len(names), desc="BLS"):

        lc, lc_flat, _phot = get_lc(
            nam,
            bundle=bundle,
            quarter=quarter,
            force_phot_column=force_phot_column,
            drop=[2],
        )
        if lc is None:
            continue

        lc_clean = lc_flat.remove_outliers(sigma_lower=1e10, sigma_upper=5)

        try:
            periodogram = lc_clean.to_periodogram(
                method="bls",
                period=period_search,
                duration=duration_search,
            )
        except:
            continue

        gaia_designation.append(lc.GAIAID)
        phot.append(_phot)
        best_periods.append(periodogram.period_at_max_power.value)
        try:
            depth_aux = periodogram.compute_stats(
                periodogram.period_at_max_power,
                periodogram.duration_at_max_power,
                periodogram.transit_time_at_max_power,
            )["depth"]
            best_depth.append(depth_aux[0].value)
            best_depth_e.append(depth_aux[1].value)
        except:
            best_depth.append(periodogram.depth_at_max_power.value)
            best_depth_e.append(np.nan)

        periods_snr.append(periodogram.snr[np.argmax(periodogram.power)].value)
        best_duration.append(periodogram.duration_at_max_power.value)
        tt0.append(periodogram.transit_time_at_max_power.value)
        power.append(periodogram.power.value)

    print(f"Total LCs with Q {quarter} {len(gaia_designation)}")
    power = np.array(power)

    bls_stats = pd.DataFrame(
        [
            gaia_designation,
            best_periods,
            best_depth,
            best_depth_e,
            periods_snr,
            best_duration,
            tt0,
            phot,
        ],
        index=[
            "gaia_designation",
            "best_period",
            "best_depth",
            "best_depth_e",
            "bls_snr",
            "best_duration",
            "best_tt0",
            "phot_type",
        ],
    ).T
    bls_stats.gaia_designation = bls_stats.gaia_designation.astype(str)
    bls_stats.bls_snr = bls_stats.bls_snr.astype(np.float64)
    bls_stats.best_depth_e = bls_stats.best_depth_e.astype(np.float64)
    bls_stats.best_depth = bls_stats.best_depth.astype(np.float64)
    bls_stats.best_period = bls_stats.best_period.astype(np.float64)
    bls_stats.best_duration = bls_stats.best_duration.astype(np.float64)
    bls_stats.best_tt0 = bls_stats.best_tt0.astype(np.float64)
    bls_stats.phot_type = bls_stats.phot_type.astype(str)

    out_dir = f"{PACKAGEDIR}/data/bls/{bundle}"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    bls_stats.to_csv(
        f"{out_dir}/bls_{bundle}_bkg_q{quarter_str}"
        f"_clean{'T' if clean_catalog else ''}_{period_range}P_bn{batch:02}.csv"
    )
    np.savez(
        f"{out_dir}/bls_{bundle}_bkg_periodogram_q{quarter_str}"
        f"_clean{'T' if clean_catalog else ''}_{period_range}P_bn{batch:02}.npz",
        power,
        period_search,
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quarter",
        dest="quarter",
        type=str,
        default=None,
        help="Quarter number.",
    )
    parser.add_argument(
        "--bundle",
        dest="bundle",
        type=str,
        default="mstars",
        help="BUndle name [mstar, wd]",
    )
    parser.add_argument(
        "--period",
        dest="period",
        type=str,
        default="short",
        help="Period range [short, long]",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        type=int,
        default=-1,
        help="Batch number if running in batch mode",
    )
    parser.add_argument(
        "--clean-catalog",
        dest="clean_catalog",
        action="store_true",
        default=False,
        help="Concatenate dir catalogs.",
    )
    args = parser.parse_args()
    try:
        args.quarter = int(args.quarter)
    except:
        pass
    do_bls_quarter(
        bundle=args.bundle,
        period_range=args.period,
        quarter=args.quarter,
        clean_catalog=args.clean_catalog,
        batch=args.batch,
    )
