import os, sys, re
import tarfile
import tempfile
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from scipy import stats
import lightkurve as lk
from scipy.stats import median_abs_deviation
import fitsio

from astropy.coordinates import SkyCoord, match_coordinates_3d
import astropy.units as u

from paths import *


def laggy_gradient(f, x=None, lag=1):
    if lag == 0:
        return f
    if x is None:
        return f[lag:] - f[:-lag]
    else:
        if f.shape != x.shape:
            raise ValueError(f"f and x have different shape {f.shape} and {x.shape}")
        return (f[lag:] - f[:-lag]) / (x[lag:] - x[:-lag])


def detrend_time_poly(time, flux, flux_err, poly_deg=3, plot=False):
    lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)

    time_norm = (time - np.nanmean(time)) / np.std(time)
    poly_dm = lk.DesignMatrix(
        np.vstack([time_norm ** idx for idx in range(poly_deg + 1)]).T, name="time_poly"
    )
    poly_dm.prior_mu = np.ones((poly_deg + 1))
    poly_dm.prior_sigma = np.ones((poly_deg + 1)) * 1e7

    rc = lk.RegressionCorrector(lc)
    clc = rc.correct(poly_dm, sigma=3, cadence_mask=None, propagate_errors=True)

    if plot:
        rc.diagnose()
        plt.show()

    return clc.flux.value


def get_lcs_from_tar_dir(dir_name, source_names, quarter=5):

    tar_name = f"{KBONUS_LCS_PATH}/kepler/{dir_name}.tar"
    print(f"Loading {len(source_names)} LCs from {tar_name}")
    tar = tarfile.open(tar_name, mode="r")
    members = tar.getnames()
    members = np.array([x for x in members if f"q{quarter:02}" in x])
    members_id = np.array([x.split("/")[1] for x in members])

    tmpdir = tempfile.TemporaryDirectory(prefix="temp_fits")

    lcs_dict = {}
    for name in source_names:
        if name in members_id:
            file_name = members[members_id == name][0]
            tar.extract(file_name, tmpdir.name)
            file_name = f"{tmpdir.name}/{file_name}"

            lc = fitsio.read(file_name, columns=["TIME", "FLUX", "FLUX_ERR"], ext=1)
            lcs_dict[name] = lc
        else:
            lcs_dict[name] = None

    return lcs_dict


def main(quarter=5, cone_dist=60):
    print("Loading KBonus catalog...")
    fname = f"{PACKAGEDIR}/data/catalogs/tpf/kbonus-bkg_kepler_v1.1.1_source_catalog_aug.csv"
    sources = pd.read_csv(fname, index_col=0)

    print("Loading KBonus quarter catalog...")
    fname = f"{PACKAGEDIR}/data/catalogs/tpf/kbonus_catalog_q{quarter:02}.csv"
    sourcesq = pd.read_csv(fname, index_col=0)

    print("Computing file names...")
    names = []
    for k, row in sourcesq.iterrows():
        if row.kic < 1.000000e20:
            names.append(int(row.kic))
        else:
            names.append(int(row.gaia_designation.split(" ")[-1]))
    names = [f"{x:09}" if x < 22934493 else str(x) for x in names]
    sourcesq["fname"] = names
    sourcesq["phot_variable_flag"] = sources.loc[
        sourcesq.gaia_designation, 'phot_variable_flag'
    ].values
    variables = sourcesq.query("phot_variable_flag == 'VARIABLE'")

    kbonus_coord = SkyCoord(
        ra=sourcesq["ra"].values,
        dec=sourcesq["dec"].values,
        frame="icrs",
        unit=(u.deg, u.deg),
    )

    nns_idx_in_cone, nns_d2d_in_cone = [], []
    contrast = []
    for k, row in tqdm(
        variables.iterrows(),
        total=len(variables),
        desc="Looking for neighbors",
    ):
        obj = SkyCoord(
            ra=[row["ra"]],
            dec=[row["dec"]],
            frame="icrs",
            unit=(u.deg, u.deg),
        )
        idxc, idxcatalog, d2d, _ = kbonus_coord.search_around_sky(
            obj, cone_dist * u.arcsec
        )

        nns_idx = idxcatalog[d2d > 0]
        nns_d2d = d2d[d2d > 0]
        _mask_contrast = row.gmag < sourcesq.gmag[nns_idx]
        nns_idx = nns_idx[_mask_contrast]
        nns_d2d = nns_d2d[_mask_contrast]
        order = np.argsort(nns_d2d)
        nns_idx = nns_idx[order]
        nns_d2d = nns_d2d[order]

        nns_idx_in_cone.append(nns_idx)
        nns_d2d_in_cone.append(nns_d2d.to("arcsec").value)

        contrast.append(sourcesq.gmag[nns_idx].values - row.gmag)

    nns_idx_in_cone = np.array(nns_idx_in_cone, dtype=object)
    nns_d2d_in_cone = np.array(nns_d2d_in_cone, dtype=object)
    contrast = np.array(contrast, dtype=object)
    nns_ids = np.unique(sourcesq.fname[np.hstack(nns_idx_in_cone)])

    pearsonr_lag_med = []
    pearsonr_lag_max = []
    pearsonr_lag_mad = []
    pearsonr = []
    lag = np.arange(10, 101)

    lcs_dict = {}

    for i, (k, row) in tqdm(
        enumerate(variables.iterrows()),
        total=len(variables),
        desc="Computing metric",
    ):

        if row.fname[:4] not in lcs_dict.keys():
            var_ids_in_dir = [x for x in variables.fname if x.startswith(row.fname[:4])]
            nns_ids_in_dir = [x for x in nns_ids if x.startswith(row.fname[:4])]
            ids_in_dir = np.unique(np.array(var_ids_in_dir + nns_ids_in_dir))

            # print("first load")
            lcs_dict[row.fname[:4]] = get_lcs_from_tar_dir(
                row.fname[:4],
                ids_in_dir,
                quarter=quarter,
            )
        else:
            var_ids_in_dir = [x for x in variables.fname if x.startswith(row.fname[:4])]
            nns_ids_in_dir = [x for x in nns_ids if x.startswith(row.fname[:4])]
            ids_in_dir = np.unique(np.array(var_ids_in_dir + nns_ids_in_dir))
            ids_in_dir = np.array(
                [x for x in ids_in_dir if x not in lcs_dict[row.fname[:4]].keys()]
            )
            if len(ids_in_dir) == 0:
                continue
            else:
                lcs_dict[row.fname[:4]].update(
                    get_lcs_from_tar_dir(
                        row.fname[:4],
                        ids_in_dir,
                        quarter=quarter,
                    )
                )

        if lcs_dict[row.fname[:4]][row.fname] is None or len(nns_idx_in_cone[i]) == 0:
            pearsonr_lag_med.append(np.ones_like(nns_idx_in_cone[i]) * np.nan)
            pearsonr_lag_max.append(np.ones_like(nns_idx_in_cone[i]) * np.nan)
            pearsonr_lag_mad.append(np.ones_like(nns_idx_in_cone[i]) * np.nan)
            pearsonr.append(np.ones_like(nns_idx_in_cone[i]) * np.nan)
            continue

        var_lc = lcs_dict[row.fname[:4]][row.fname]

        if not np.isfinite(var_lc["FLUX"]).all():
            pearsonr_lag_med.append(np.ones_like(nns_idx_in_cone[i]) * np.nan)
            pearsonr_lag_max.append(np.ones_like(nns_idx_in_cone[i]) * np.nan)
            pearsonr_lag_mad.append(np.ones_like(nns_idx_in_cone[i]) * np.nan)
            pearsonr.append(np.ones_like(nns_idx_in_cone[i]) * np.nan)
            continue

        var_flux = detrend_time_poly(var_lc["TIME"], var_lc["FLUX"], var_lc["FLUX_ERR"])
        var_flux_norm = var_flux / np.nanmean(var_flux)

        nns_dirs = np.unique([x[:4] for x in sourcesq.fname[nns_idx_in_cone[i]]])
        # print(nns_dirs)
        for ds in nns_dirs:
            if ds not in lcs_dict.keys():
                nns_ids_in_dir = [x for x in nns_ids if x.startswith(ds)]
                # print("second load")
                lcs_dict[ds] = get_lcs_from_tar_dir(
                    ds,
                    np.unique(nns_ids_in_dir),
                    quarter=quarter,
                )
            else:
                nns_ids_in_dir = [x for x in nns_ids if x.startswith(ds)]
                nns_ids_in_dir = [
                    x for x in nns_ids_in_dir if x not in lcs_dict[ds].keys()
                ]
                if len(nns_ids_in_dir) == 0:
                    continue
                # print("third load")
                lcs_dict[ds].update(
                    get_lcs_from_tar_dir(
                        ds,
                        np.unique(nns_ids_in_dir),
                        quarter=quarter,
                    )
                )
        # print(lcs_dict.keys())

        aux_p_med = []
        aux_p_max = []
        aux_p_mad = []
        aux_p = []

        for n, idx in enumerate(nns_idx_in_cone[i]):

            if lcs_dict[sourcesq.fname[idx][:4]][sourcesq.fname[idx]] is None:
                aux_p_med.append(np.nan)
                aux_p_max.append(np.nan)
                aux_p_mad.append(np.nan)
                aux_p.append(np.nan)
                continue

            nn_lc = lcs_dict[sourcesq.fname[idx][:4]][sourcesq.fname[idx]]

            if not np.isfinite(nn_lc["FLUX"]).any():
                aux_p_med.append(np.nan)
                aux_p_max.append(np.nan)
                aux_p_mad.append(np.nan)
                aux_p.append(np.nan)
                continue

            nn_flux = detrend_time_poly(nn_lc["TIME"], nn_lc["FLUX"], nn_lc["FLUX_ERR"])
            nn_flux_norm = nn_lc["FLUX"] / np.nanmean(nn_lc["FLUX"])

            if nn_flux.shape != var_flux.shape:
                aux_p_med.append(np.nan)
                aux_p_max.append(np.nan)
                aux_p_mad.append(np.nan)
                aux_p.append(np.nan)
                continue

            pearsonr_ = np.zeros(lag.shape, dtype=float)
            pearsonp_ = np.zeros(lag.shape, dtype=float)
            for n, val in enumerate(lag):
                laggrad_var = laggy_gradient(var_flux_norm, var_lc["TIME"], lag=val)
                laggrad_obj = laggy_gradient(nn_flux_norm, var_lc["TIME"], lag=val)
                pearsonr_[n], pearsonp_[n] = stats.pearsonr(laggrad_var, laggrad_obj)

            aux_p_med.append(np.nanmedian(pearsonr_))
            aux_p_max.append(np.nanmax(np.abs(pearsonr_)))
            aux_p_mad.append(median_abs_deviation(pearsonr_))
            aux_p.append(stats.pearsonr(nn_flux_norm, var_flux_norm)[0])

        pearsonr_lag_med.append(np.array(aux_p_med))
        pearsonr_lag_max.append(np.array(aux_p_max))
        pearsonr_lag_mad.append(np.array(aux_p_mad))
        pearsonr.append(np.array(aux_p))
        # break

    pearsonr_lag_med = np.asarray(pearsonr_lag_med, dtype=object)
    pearsonr_lag_max = np.asarray(pearsonr_lag_max, dtype=object)
    pearsonr_lag_mad = np.asarray(pearsonr_lag_mad, dtype=object)
    pearsonr = np.asarray(pearsonr, dtype=object)

    metrics_all = pd.DataFrame(
        np.vstack(
            [
                np.hstack(nns_d2d_in_cone),
                np.hstack(contrast),
                np.hstack([x.T for x in pearsonr_lag_med if x.shape != (0,)]),
                np.hstack([x.T for x in pearsonr_lag_max if x.shape != (0,)]),
                np.hstack([x.T for x in pearsonr_lag_mad if x.shape != (0,)]),
                np.hstack([x.T for x in pearsonr if x.shape != (0,)]),
            ]
        ).T,
        index=[
            f"{k}_{j}" for k, sublist in enumerate(nns_idx_in_cone) for j in sublist
        ],
        columns=[
            "dist",
            "contrast",
            "pearson_lag_med",
            "pearson_lag_max",
            "pearson_lag_mad",
            "pearson",
        ],
    )

    metrics_all["gaia_designation"] = sources.iloc[
        [int(x.split("_")[-1]) for x in metrics_all.index]
    ].index
    metrics_all["kic"] = sources.iloc[
        [int(x.split("_")[-1]) for x in metrics_all.index]
    ].kic.values
    metrics_all["gaia_corr_origen"] = variables.iloc[
        [int(x.split("_")[0]) for x in metrics_all.index]
    ].gaia_designation.values
    metrics_all["kic_corr_origen"] = variables.iloc[
        [int(x.split("_")[0]) for x in metrics_all.index]
    ].kic.values

    csv_name = f"{PACKAGEDIR}/data/catalogs/tpf/correlation_metrics_q{quarter:02}.csv"
    print(f"Saving metric table in \n {csv_name}")
    metrics_all.to_csv(csv_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoEncoder")
    parser.add_argument(
        "--quarter",
        dest="quarter",
        type=int,
        default=None,
        help="Quarter number.",
    )
    parser.add_argument("--log", dest="log", default=0, help="Logging level")
    args = parser.parse_args()

    main(quarter=args.quarter)
