import os, re
import glob
import tarfile
import tempfile
import subprocess
import feets
import yaml
import numpy as np
import pandas as pd
import lightkurve as lk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sb
from astropy.io import fits
from tqdm.auto import tqdm
import astropy.units as u
from astropy.stats import sigma_clip
from sklearn import linear_model
from astropy.coordinates import SkyCoord, match_coordinates_3d
import astropy.units as u
from paths import *

# kepler_root_dir = "/Users/jorgemarpa/Work/BAERI/ADAP/data/kepler"
kepler_root_dir = "/Volumes/jorge-marpa/Work/BAERI/data/kepler/"
qd_map = {
    1: 2009166043257,
    2: 2009259160929,
    3: 2009350155506,
    4: 2010078095331,
    5: 2010174085026,
    6: 2010265121752,
    7: 2010355172524,
    8: 2011073133259,
    9: 2011177032512,
    10: 2011271113734,
    11: 2012004120508,
    12: 2012088054726,
    13: 2012179063303,
    14: 2012277125453,
    15: 2013011073258,
    16: 2013098041711,
    17: 2013131215648,
}


def get_archive_lightcurves(tar_files):
    if isinstance(tar_files, str):
        tar_files = [tar_files]

    lcs, kics, tpf_org = [], [], []
    for i, tar in enumerate(tar_files):
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(tar, mode="r:gz") as tar:
                tar.extractall(path=tmpdir)
            archive = glob.glob(f"{tmpdir}/*.fits")

            for f in tqdm(archive, desc=f"Loading from tarball {i+1}/{len(tar_files)}"):
                lc = lk.KeplerLightCurve.read(f)
                lcs.append(lc)
                try:
                    kics.append(f"{lc.meta['KEPLERID']:09}")
                except:
                    kics.append(None)
                try:
                    tpf_org.append(f"{lc.meta['TPFORG']:09}")
                except:
                    tpf_org.append(None)

    return lcs, kics, tpf_org


def get_keple_lightcurves(kics, quarter, tar=True):

    lcs = []
    exist = False
    if tar:
        with tempfile.TemporaryDirectory() as tmpdir:
            for kic in tqdm(kics, desc="Loading Kepler LCs"):
                # skip None IDs
                if kic is None:
                    lcs.append(None)
                    continue
                tarname = glob.glob(
                    f"{kepler_root_dir}/lcs/{kic[:4]}/kplr{kic}_lc_Q*.tar"
                )
                # skip when tarball doesn't exist
                if len(tarname) == 0:
                    lcs.append(None)
                    continue
                else:
                    tarname = tarname[0]
                to_extract = f"{kic}/kplr{kic}-{qd_map[quarter]}_llc.fits"
                # unpack
                with tarfile.open(tarname, mode="r") as tar:
                    try:
                        tar.extract(to_extract, path=tmpdir)
                    # skip when quarter doesn’t exist
                    except KeyError:
                        lcs.append(None)
                        continue

                fin = f"{tmpdir}/{to_extract}"

                if os.path.isfile(fin):
                    lcs.append(lk.KeplerLightCurve.read(fin))
                    exist = True
                else:
                    lcs.append(None)
                    continue
    else:
        for kic in tqdm(kics, desc="Loading Kepler LCs"):
            # skip None IDs
            if kic is None:
                lcs.append(None)
                continue
            fname = (
                f"{ARCHIVE_PATH}/data/kepler/lcs"
                f"/{kic[:4]}/kplr{kic}-{qd_map[quarter]}_llc.fits"
            )
            if os.path.isfile(fname):
                lcs.append(lk.KeplerLightCurve.read(fname))
                exist = True
            else:
                lcs.append(None)
                continue

    return lcs, exist


def make_lc_download_sh(unique_kics, channel, quarter):

    sh_name = f"../bash/kepler_lcs_ch{channel:02}_q{quarter:02}.sh"
    try:
        os.remove(sh_name)
    except FileNotFoundError:
        pass
    tar_file_list = pd.read_csv(
        "/Users/jorgemarpa/Work/BAERI/ADAP/kepler-workflow"
        + "/data/support/kepler_lc_tars_path.txt",
        header=None,
        names=["dir1", "dir2", "tarname"],
        delimiter="/",
    )
    with open(sh_name, "a") as f:
        f.write("#!/bin/bash\n")
        for j, kic in enumerate(unique_kics):
            if kic is None:
                continue

            df_idx = np.where(int(kic) == tar_file_list.dir2)[0]
            if len(df_idx) == 0:
                continue
            else:
                df_idx = df_idx[0]
            tarname = tar_file_list.loc[df_idx, "tarname"]
            dir_name = f"{kepler_root_dir}/lcs/{kic[:4]}"
            fout = f"{dir_name}/{tarname}"
            if os.path.isfile(fout):
                continue
            line = (
                f"curl --globoff --location-trusted -f --progress --create-dirs --output "
                f"'{fout}' "
                f"'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:Kepler/url/"
                f"missions/kepler/lightcurves/"
                f"{kic[:4]}/{kic}/{tarname}' &\n"
            )
            f.write(line)
            if j % 10 == 0 and j > 0:
                f.write("process_id=$!\n")
                f.write("wait $process_id\n")
    return sh_name


# for kid in tqdm(unique_kics):
#     kic = f"{kid:09}"
#     url = (f"https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:Kepler/url/"
#            f"missions/kepler/lightcurves/{kic[:4]}/{kic}/kplr{kic}-2010174085026_llc.fits")
#     print(url)
#     wget.download(url, out="lcs/Kepler/")


def compute_stats_from_lcs(lcs, project="kbonus", do_cdpp=True):
    mean_lcs_ap1, mean_lcs_ap2, = (
        [],
        [],
    )
    cdpp_lcs_ap1, cdpp_lcs_ap2, cdpp_lcs_ap3 = [], [], []
    cat_mag = []
    ra_lcs, dec_lcs = [], []
    complet, contam = [], []

    for i in tqdm(
        range(len(lcs)), total=len(lcs), desc=f"Computing stats for {project}"
    ):

        # skip None lcs or lcs with all NaN
        if not lcs[i] or (
            np.isnan(lcs[i].flux.value).all() and np.isnan(lcs[i].sap_flux.value).all()
        ):
            ra_lcs.append(np.nan)
            dec_lcs.append(np.nan)
            cat_mag.append(np.nan)
            mean_lcs_ap1.append(np.nan)
            mean_lcs_ap2.append(np.nan)
            cdpp_lcs_ap1.append(np.nan)
            cdpp_lcs_ap2.append(np.nan)
            cdpp_lcs_ap3.append(np.nan)
            complet.append(np.nan)
            contam.append(np.nan)
            continue

        lc = lcs[i].copy()
        ra_lcs.append(lc.ra)
        dec_lcs.append(lc.dec)

        if lc.flux.unit != "electron / s" and project == "kbonus":
            lc.flux = lc.flux.value * (u.electron / u.second)
            lc.flux_err = lc.flux_err.value * (u.electron / u.second)
            lc.sap_flux = lc.sap_flux.value * (u.electron / u.second)
            lc.sap_flux_err = lc.sap_flux_err.value * (u.electron / u.second)
        # print(lc)

        if project == "kbonus":
            cat_mag.append(lc.meta["GMAG"])
            mean_lcs_ap1.append(np.nanmean(lc.flux).value)
            mean_lcs_ap2.append(np.nanmean(lc.sap_flux).value)
            # CDPP values
            if do_cdpp:
                if (lc.flux == 0).all() or not np.isfinite(lc.flux).all():
                    cdpp_lcs_ap1.append(np.nan)
                else:
                    try:
                        cdpp_lcs_ap1.append(lc.estimate_cdpp().value)
                    except:
                        cdpp_lcs_ap1.append(np.nan)

                if (lc.sap_flux == 0).all():
                    cdpp_lcs_ap2.append(np.nan)
                else:
                    lc.flux = lc.sap_flux
                    lc.flux_err = lc.sap_flux_err
                    try:
                        cdpp_lcs_ap2.append(lc.estimate_cdpp().value)
                    except:
                        cdpp_lcs_ap2.append(np.nan)
                if (lc.psf_flux_nvs == 0).all():
                    cdpp_lcs_ap3.append(np.nan)
                else:
                    lc.flux = lc.psf_flux_nvs
                    lc.flux_err = lc.psf_flux_err_nvs
                    try:
                        cdpp_lcs_ap3.append(lc.estimate_cdpp().value)
                    except:
                        cdpp_lcs_ap3.append(np.nan)
            else:
                cdpp_lcs_ap1.append(np.nan)
                cdpp_lcs_ap2.append(np.nan)
                cdpp_lcs_ap3.append(np.nan)

        else:
            cat_mag.append(lc.meta["KEPMAG"])
            mean_lcs_ap1.append(np.nanmean(lc.pdcsap_flux).value)
            mean_lcs_ap2.append(np.nanmean(lc.sap_flux).value)
            # CDPP values
            if do_cdpp:
                lc.flux = lc.pdcsap_flux
                lc.flux_err = lc.pdcsap_flux_err
                cdpp_lcs_ap1.append(lc.estimate_cdpp().value)
                lc.flux = lc.sap_flux
                lc.flux_err = lc.sap_flux_err
                cdpp_lcs_ap2.append(lc.estimate_cdpp().value)
            else:
                cdpp_lcs_ap1.append(np.nan)
                cdpp_lcs_ap2.append(np.nan)

        complet.append(float(lc.FLFRCSAP) if lc.FLFRCSAP != "" else np.nan)
        contam.append(float(lc.CROWDSAP) if lc.CROWDSAP != "" else np.nan)

        # break
    mean_lcs_ap1 = np.array(mean_lcs_ap1)
    mean_lcs_ap2 = np.array(mean_lcs_ap2)
    cdpp_lcs_ap1 = np.array(cdpp_lcs_ap1)
    cdpp_lcs_ap2 = np.array(cdpp_lcs_ap2)
    cat_mag = np.array(cat_mag)
    ra_lcs = np.array(ra_lcs)
    dec_lcs = np.array(dec_lcs)

    if project == "kbonus":
        cdpp_lcs_ap3 = np.array(cdpp_lcs_ap3)
        return {
            "lc_mean_psf": mean_lcs_ap1,
            "lc_mean_sap": mean_lcs_ap2,
            "lc_cdpp_psf": cdpp_lcs_ap1,
            "lc_cdpp_sap": cdpp_lcs_ap2,
            "lc_cdpp_psf_nva": cdpp_lcs_ap3,
            "g_mag": cat_mag,
            "ra": ra_lcs,
            "dec": dec_lcs,
            "FLFRCSAP": complet,
            "CROWDSAP": contam,
        }
    else:
        return {
            "lc_mean_pdc": mean_lcs_ap1,
            "lc_mean_sap": mean_lcs_ap2,
            "lc_cdpp_pdc": cdpp_lcs_ap1,
            "lc_cdpp_sap": cdpp_lcs_ap2,
            "kep_mag": cat_mag,
            "ra": ra_lcs,
            "dec": dec_lcs,
            "FLFRCSAP": complet,
            "CROWDSAP": contam,
        }


def compute_zero_point(X, y, use_ransac=True):
    if use_ransac:
        ransac = linear_model.RANSACRegressor()
        # remove nans
        nan_mask = np.isfinite(X) & np.isfinite(y)
        ransac.fit(X[nan_mask, None], y[nan_mask, None])
        factor = ransac.estimator_.coef_[0, 0]

        # Predict data of estimated models
        line_X = np.arange(np.nanmin(X), np.nanmax(X))[:, np.newaxis]
        line_y = ransac.predict(line_X)
        return factor, line_X, line_y, ransac

    else:
        ratio = y / X
        reject_mask = sigma_clip(ratio, sigma=5).mask
        factor = np.median(ratio[~reject_mask])

        line_X = np.arange(np.nanmin(X), np.nanmax(X))
        line_y = line_X * factor
        return factor, line_X, line_y, None


def plot_joint(kp_stats, jm_stats, meta):
    def scatter_hist(
        x, y, ax, ax_histx, ax_histy, c="k", label="KepPipe", ylim=[0, 1e3]
    ):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        x = x[(y >= ylim[0]) & (y <= ylim[1])]
        y = y[(y >= ylim[0]) & (y <= ylim[1])]

        # the scatter plot:
        ax.scatter(x, y, c=c, s=2, alpha=0.3, label=label)

        # ax_histx.hist(x, bins=40, color=c, histtype="step", lw=2, alpha=.8)
        sb.kdeplot(x, color=c, lw=2, alpha=0.8, ax=ax_histx)
        # ax_histy.hist(y, bins=40, orientation='horizontal', range=[0, 6e2], color=c,
        # histtype="step", lw=2, alpha=.8)

        sb.kdeplot(y, color=c, lw=2, alpha=0.8, ax=ax_histy, vertical=True)

    ylim = [-5, 6e2]
    quarter = meta["quarter"]
    channel = meta["channel"]

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"Kepler \nQ{quarter} Ch{channel}", x=0.8, y=0.85, fontsize=20)

    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(7, 2),
        height_ratios=(2, 7),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.05,
    )

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    if isinstance(kp_stats, dict):
        mask = np.isfinite(kp_stats["lc_cdpp_pdc"]) & np.isfinite(
            kp_stats["lc_mean_pdc"]
        )
        scatter_hist(
            np.log10(kp_stats["lc_mean_pdc"][mask]),
            kp_stats["lc_cdpp_pdc"][mask],
            ax,
            ax_histx,
            ax_histy,
            c="k",
            label="KepPipe",
            ylim=ylim,
        )

    mask = np.isfinite(jm_stats["lc_cdpp_sap"]) & np.isfinite(jm_stats["lc_mean_sap"])
    scatter_hist(
        np.log10(jm_stats["lc_mean_sap"][mask]),
        jm_stats["lc_cdpp_sap"][mask],
        ax,
        ax_histx,
        ax_histy,
        c="tab:blue",
        label="KBonus_SAP",
        ylim=ylim,
    )

    mask = np.isfinite(jm_stats["lc_cdpp_psf"]) & np.isfinite(
        jm_stats["lc_mean_psf_zp"]
    )
    scatter_hist(
        np.log10(jm_stats["lc_mean_psf_zp"][mask]),
        jm_stats["lc_cdpp_psf"][mask],
        ax,
        ax_histx,
        ax_histy,
        c="tab:orange",
        label="KBonus_PSF",
        ylim=ylim,
    )

    ax.set_xlim(2.8, 6)
    ax.set_ylim(ylim)
    ax.set_xlabel("log(<flux>)", fontsize=15)
    ax.set_ylabel("6.5h-CDPP [ppm]", fontsize=15)
    ax.legend(loc="lower left", fontsize=15, markerscale=5)

    plt.show()


def get_features(lcs, flux_col="flux"):

    feat_list = [
        "FluxPercentileRatioMid20",
        "FluxPercentileRatioMid50",
        "FluxPercentileRatioMid80",
        "PercentDifferenceFluxPercentile",
        "LinearTrend",
        "Amplitude",
        "Rcs",
        "Q31",
    ]
    fs = feets.FeatureSpace(only=feat_list, data=["magnitude", "time"])

    results = {}
    for k, lc in tqdm(
        enumerate(lcs), total=len(lcs), desc=f"Computing Features for {flux_col}"
    ):
        if lc is None:
            results[k] = np.array([np.nan] * len(feat_list))
            continue
        flux = lc[flux_col].value / np.nanmedian(lc[flux_col].value)
        mask = np.isfinite(flux)
        if (~mask).sum() == mask.shape[0]:
            results[k] = np.array([np.nan] * len(feat_list))
            continue
        if flux_col == "sap_flux" and (flux[mask] == 0).all():
            results[k] = np.array([np.nan] * len(feat_list))
            continue
        lc = np.array([lc.time.value[mask], flux[mask]])
        features, values = fs.extract(*lc)
        results[k] = values
        # if k ==10: break
    results = pd.DataFrame.from_dict(results, columns=features, orient="index")

    return results


def find_lc_examples(jm_stats, lcs):
    jm_stats_df = pd.DataFrame.from_dict(jm_stats)
    flux_bright = [1e5, 1e6]
    flux_faint = [1e2, 5e3]

    brights = jm_stats_df.query(
        f"lc_mean_psf_zp > {flux_bright[0]} and lc_mean_psf_zp < {flux_bright[1]}"
    )

    faints = jm_stats_df.query(
        f"lc_mean_psf_zp > {flux_faint[0]} and lc_mean_psf_zp < {flux_faint[1]}"
    )

    cdpp_bright = [
        np.percentile(brights["lc_cdpp_psf"], 15),
        np.percentile(brights["lc_cdpp_psf"], 85),
    ]
    cdpp_faint = [
        np.percentile(faints["lc_cdpp_psf"], 15),
        np.percentile(faints["lc_cdpp_psf"], 85),
    ]

    bright_goods = brights.query(f"lc_cdpp_psf < {cdpp_bright[0]}").index.values
    bright_bads = brights.query(f"lc_cdpp_psf > {cdpp_bright[1]}").index.values
    faint_goods = faints.query(f"lc_cdpp_psf < {cdpp_faint[0]}").index.values
    faint_bads = faints.query(f"lc_cdpp_psf > {cdpp_faint[1]}").index.values

    # find neighbors
    sources = SkyCoord(
        ra=jm_stats["ra"],
        dec=jm_stats["dec"],
        frame="icrs",
        unit=(u.deg, u.deg),
    )
    midx, mdist = match_coordinates_3d(sources, sources, nthneighbor=2)[:2]
    dist_mask = mdist.arcsec < 4
    # ged delta mag between close neighbors
    neighbors_dm = np.abs(
        jm_stats["g_mag"][np.arange(len(sources))[dist_mask]]
        - jm_stats["g_mag"][midx[dist_mask]]
    )
    mag_mask = neighbors_dm < 0.5
    # finf pair of LC index that correspond to neighbors of similar birghtness
    idx_cont1 = np.arange(len(sources))[dist_mask][mag_mask]
    # idx_cont1 = [i for i in idx_cont1 if np.isfinite(lcs[i].flux).all()]
    idx_cont2 = midx[dist_mask][mag_mask]
    # idx_cont2 = [i for i in idx_cont2 if np.isfinite(lcs[i].flux).all()]
    drop = []
    for k in range(len(idx_cont1)):
        if (
            np.isfinite(lcs[idx_cont1[k]].flux).all()
            or np.isfinite(lcs[idx_cont2[k]].flux).all()
        ):
            continue
        else:
            drop.append(k)
    idx_cont1 = np.delete(idx_cont1, drop)
    idx_cont2 = np.delete(idx_cont2, drop)

    rnd_cont = np.random.randint(0, len(idx_cont1))

    lc_ex_idx = [
        np.random.choice(bright_goods),
        np.random.choice(faint_goods),
        np.random.choice(bright_bads),
        np.random.choice(faint_bads),
        idx_cont1[rnd_cont],
        idx_cont2[rnd_cont],
    ]

    return lc_ex_idx


def make_dashboard(stats, features, lightcurves, meta, save=True, name=None):

    if name:
        try:
            bkg = True if name[re.search("_bkg", name).span()[1]] == "T" else False
        except:
            bkg = False
        try:
            fva = True if name[re.search("_fva", name).span()[1]] == "T" else False
        except:
            fva = True
        try:
            augment = True if name[re.search("_aug", name).span()[1]] == "T" else False
        except:
            augment = bkg
        try:
            sgmt = True if name[re.search("_sgm", name).span()[1]] == "T" else False
        except:
            sgmt = False

    with open(
        "%s/kepler_workflow/config/tpfmachine_keplerTPFs_config.yaml" % (PACKAGEDIR),
        "r",
    ) as f:
        config = yaml.safe_load(f)
    config["fit_va"] = fva
    config["fit_bkg"] = bkg
    config["augment_bkg"] = augment
    config["segments"] = sgmt

    df = pd.DataFrame.from_dict([config]).T.reset_index(drop=False)
    df = df.rename({"index": "Parameter", 0: "value"}, axis=1)

    color = [["w"] * df.shape[1] for k in range(df.shape[0])]
    for k in range(len(color)):
        if isinstance(df.values[k][1], bool):
            color[k][1] = "tab:green" if df.values[k][1] else "tab:red"

    channel = meta["channel"]
    quarter = meta["quarter"]
    psf_zp = meta["psf_zp"]
    sap_zp = meta["sap_zp"]

    feat_kp_pdc = features["feat_kp_pdc"]
    feat_kp_sap = features["feat_kp_sap"]
    feat_jm_sap = features["feat_jm_sap"]
    feat_jm_psfnv = features["feat_jm_psfnv"]
    feat_jm_psf = features["feat_jm_psf"]

    lcs = lightcurves["lcs"]
    kplcs = lightcurves["kplcs"]

    kp_stats = stats["kp_stats"]
    jm_stats = stats["jm_stats"]

    lc_ex_idx = find_lc_examples(jm_stats, lcs)

    fig = plt.figure(figsize=(24, 36))
    fig.suptitle(
        f"Kepler Q{quarter} Ch{channel} N {len(lcs):,}", x=0.5, y=0.895, fontsize=20
    )
    G = gridspec.GridSpec(9, 4, figure=fig)

    fontsize = 10
    markerscale = 5
    lw = 1.5
    ms = 1.2

    ##################################################################################
    ##################################################################################

    ax_table1 = fig.add_subplot(G[0, 2])
    ax_table2 = fig.add_subplot(G[0, 3])

    ax_table1.axis("off")
    ax_table1.axis("tight")
    ax_table1.table(
        cellText=df.values[: df.shape[0] // 2],
        colLabels=None,
        loc="center",
        cellColours=color[: df.shape[0] // 2],
        fontsize=10,
    )
    # ax_table1.set_fontsize(14)
    # ax_table1.scale(1.2, 1.2)

    ax_table2.axis("off")
    ax_table2.axis("tight")
    ax_table2.table(
        cellText=df.values[df.shape[0] // 2 :],
        colLabels=None,
        loc="center",
        cellColours=color[df.shape[0] // 2 :],
        fontsize=10,
    )
    # ax_table2.set_fontsize(14)
    # ax_table2.scale(1.2, 1.2)

    ##################################################################################
    ##################################################################################

    ax_00 = fig.add_subplot(G[1, 0])

    if isinstance(kp_stats, dict):
        ax_00.scatter(
            kp_stats["lc_mean_pdc"],
            kp_stats["lc_mean_pdc"] / jm_stats["lc_mean_psf"],
            s=ms,
            label=f"KP/psf = {psf_zp:0.4f}",
            alpha=0.7,
            c="tab:orange",
        )
        ax_00.axhline(psf_zp, c="tab:orange", ls="-")
        ax_00.scatter(
            kp_stats["lc_mean_pdc"],
            kp_stats["lc_mean_pdc"] / jm_stats["lc_mean_sap"],
            s=ms,
            label=f"KP/sap = {sap_zp:0.4f}",
            alpha=0.7,
            c="tab:blue",
        )
        ax_00.axhline(sap_zp, c="tab:blue", ls="-")
        ax_00.set_xscale("log")
        ax_00.set_xlabel("log(<flux>) Kepler Pipeline")
        ax_00.set_ylabel("Kepler Pipeline / KBonus")
        ax_00.set_ylim(0, 2.5)
        ax_00.legend(loc="upper right", markerscale=markerscale, fontsize=fontsize)
    else:
        ax_00.plot([1e3, 1e7], [1e3, 1e7], c="tab:red", ls="-")
        ax_00.scatter(
            jm_stats["lc_mean_sap"],
            jm_stats["lc_mean_psf"],
            s=ms,
            alpha=0.7,
            c="k",
        )
        ax_00.set_xscale("log")
        ax_00.set_yscale("log")
        ax_00.set_xlabel("log(<flux>) SAP")
        ax_00.set_ylabel("log(<flux>) PSF")
        ax_00.set_xlim(1e3, 1e7)
        ax_00.set_ylim(1e3, 1e7)

    ##################################################################################
    ##################################################################################

    ylim = [-7.5, 1e3]
    xlim = [2.2, 6]
    ax_01 = fig.add_subplot(G[1, 1])
    ax_02 = fig.add_subplot(G[1, 2])
    ax_03 = fig.add_subplot(G[1, 3])
    kde_axis = [ax_01, ax_02, ax_03]
    # gs02 = G[0, 2].subgridspec(2, 1)
    # gs02ax0 = fig.add_subplot(gs02[0])
    # gs02ax1 = fig.add_subplot(gs02[1])

    ax_10 = fig.add_subplot(G[2, 0])
    ax_11 = fig.add_subplot(G[2, 1])

    if isinstance(kp_stats, dict):
        colors = ["k", "tab:blue", "tab:orange"]
        labels = ["Kepler PDC", "KBonus SAP", "KBonus PSF"]
        Xs = [
            kp_stats["lc_mean_pdc"],
            jm_stats["lc_mean_sap"],
            jm_stats["lc_mean_psf_zp"],
        ]
        Ys = [kp_stats["lc_cdpp_pdc"], jm_stats["lc_cdpp_sap"], jm_stats["lc_cdpp_psf"]]
    else:
        colors = ["k", "tab:blue", "tab:orange"]
        labels = ["Kepler Pipeline", "KBonus SAP", "KBonus PSF"]
        Xs = [None, jm_stats["lc_mean_sap"], jm_stats["lc_mean_psf_zp"]]
        Ys = [None, jm_stats["lc_cdpp_sap"], jm_stats["lc_cdpp_psf"]]

    for k in range(len(Xs)):
        x = Xs[k]
        y = Ys[k]
        if isinstance(x, np.ndarray):
            mask = (y >= ylim[0]) & (y <= ylim[1]) & (x < 1e6)
            x = np.log10(x[mask])
            y = y[mask]

            mask = np.isfinite(x) & np.isfinite(y)
            kde_axis[k].scatter(x, y, c=colors[k], marker=".", s=ms, label=labels[k])
            sb.histplot(
                x=x,
                y=y,
                color=colors[k],
                pthresh=0.05,
                bins="auto",
                ax=kde_axis[k],
                cbar=True,
            )
            sb.kdeplot(
                x[mask], color=colors[k], lw=lw, alpha=0.8, ax=ax_10, label=labels[k]
            )
            sb.kdeplot(y[mask], color=colors[k], lw=lw, alpha=0.8, ax=ax_11)
            kde_axis[k].legend(
                loc="lower left", fontsize=fontsize, markerscale=markerscale
            )
        else:
            pass

        kde_axis[k].set_xlim(xlim)
        kde_axis[k].set_ylim(ylim)
        kde_axis[k].set_xlabel("log(<Flux>)", fontsize=fontsize)
        kde_axis[k].set_ylabel("6.5h-CDPP [ppm]", fontsize=fontsize)

    ax_10.set_xlabel("log(<Flux>)", fontsize=fontsize)
    ax_11.set_xlabel("6.5h-CDPP [ppm]", fontsize=fontsize)

    ##################################################################################
    ##################################################################################

    ax_11 = fig.add_subplot(G[2, 2])
    (feat_jm_sap["LinearTrend"]).plot(
        kind="kde", ax=ax_11, label="KBonus_SAP", color="tab:blue"
    )
    (feat_jm_psfnv["LinearTrend"]).plot(
        kind="kde", ax=ax_11, label="KBonus_PSFNV", color="tab:green"
    )
    (feat_jm_psf["LinearTrend"]).plot(
        kind="kde", ax=ax_11, label="KBonus_PSF", color="tab:orange"
    )
    ax_11.set_xlim(-0.5e-1, 0.5e-1)
    ax_11.set_xlabel("LinearTrend")
    ax_11.legend(loc="upper left")

    ax_12 = fig.add_subplot(G[2, 3])
    (feat_jm_sap["FluxPercentileRatioMid20"]).plot(
        kind="kde", ax=ax_12, label="_nolegend_", color="tab:blue"
    )
    (feat_jm_psfnv["FluxPercentileRatioMid20"]).plot(
        kind="kde", ax=ax_12, label="_nolegend_", color="tab:green"
    )
    (feat_jm_psf["FluxPercentileRatioMid20"]).plot(
        kind="kde", ax=ax_12, label="_nolegend_", color="tab:orange"
    )
    ax_12.axvline(0.154, c="tab:red", lw=lw / 2, label="Expected for Flux ~ N(0,1)")
    # ax_12.set_xlim(-1.5, 0.5)
    ax_12.set_xlabel(r"$(F_{60} - F_{40}) / (F_{95} - F_{5})$")
    ax_12.legend(loc="upper right")

    if isinstance(feat_kp_pdc, pd.DataFrame):
        (feat_kp_pdc["LinearTrend"]).plot(
            kind="kde", ax=ax_11, label="KepPipe_PDC", color="k"
        )
        (feat_kp_sap["LinearTrend"]).plot(
            kind="kde", ax=ax_11, label="KepPipe_SAP", color="gray", ls="--"
        )
        (feat_kp_pdc["FluxPercentileRatioMid20"]).plot(
            kind="kde", ax=ax_12, label="_nolegend_", color="k"
        )
        (feat_kp_sap["FluxPercentileRatioMid20"]).plot(
            kind="kde", ax=ax_12, label="_nolegend_", color="gray", ls="--"
        )

        ax_11.set_xlim(-1e-2, 1e-2)
        ax_11.set_ylim(-10, ax_11.get_ylim()[1] / 10)
        ax_11.xaxis.set_major_locator(plt.MaxNLocator(5))

    for i, k in enumerate(lc_ex_idx):
        lc = lcs[k].copy()
        if lc.time.value[0] > 2454833:
            lc.time = lc.time - 2454833
        ax_20 = fig.add_subplot(G[3 + i, 0])
        tpf = lk.search_targetpixelfile(
            f"KIC {lc.TPFORG:09}",
            quarter=lc.QUARTER,
            mission=lc.MISSION,
            cadence="long",
        ).download()
        tpf.plot(
            frame=100,
            ax=ax_20,
            title="",
            scale="log",
        )
        # add background sources
        col, row = tpf.wcs.all_world2pix(
            np.array([jm_stats["ra"], jm_stats["dec"]]).T, 0
        ).T
        col += tpf.column
        row += tpf.row
        ax_20.scatter(col, row, marker="x", c="tab:red")

        # add target
        ax_20.scatter(
            col[k],
            row[k],
            marker="o",
            c="tab:red",
        )

        ax_20.set_xlim(
            np.minimum(tpf.column, col[k]) - 1,
            np.maximum(tpf.column + tpf.shape[-1], col[k] + 0.5),
        )
        ax_20.set_ylim(
            np.minimum(tpf.row, row[k]) - 1,
            np.maximum(tpf.row + tpf.shape[-2], row[k] + 0.5),
        )

        ax_21 = fig.add_subplot(G[3 + i, 1:])

        if isinstance(kplcs, list) and kplcs[k] is not None:
            klc = kplcs[k].copy()
            klc.normalize().plot(label="Kepler Pipeline PDC", c="k", ax=ax_21)
            klc.flux = klc.sap_flux
            klc.flux_err = klc.sap_flux_err
            klc.normalize().plot(label="Kepler Pipeline SAP", c="gray", ax=ax_21)
        else:
            pass

        lc.normalize().plot(ax=ax_21, c="tab:orange", label="KBonus PSF")

        lc.flux = lc.sap_flux
        lc.flux_err = lc.sap_flux_err
        lc.normalize().plot(ax=ax_21, c="tab:blue", label="KBonus SAP")

        lc.flux = lc.psf_flux_nvs
        lc.flux_err = lc.psf_flux_err_nvs
        lc.normalize().plot(ax=ax_21, c="tab:green", label="KBonus PSF-NV")

        txt = (
            f"{lc.GAIAID} / KIC {lc.KEPLERID}\n"
            f"G MAG         : {jm_stats['g_mag'][k]:.3f}\n"
            f"FLFRCSAP    : {jm_stats['FLFRCSAP'][k]:.3f}\n"
            f"CROWDSAP : {jm_stats['CROWDSAP'][k]:.3f}\n"
            f"CDPP (PSF)  : {jm_stats['lc_cdpp_psf'][k]:.3f}\n"
            f"CDPP (SAP)  : {jm_stats['lc_cdpp_sap'][k]:.3f}"
        )
        if isinstance(kp_stats, dict):
            txt = f"{txt}\n" f"CDPP (PDC)  : {kp_stats['lc_cdpp_pdc'][k]:.3f}"
        ax_21.text(
            0.3,
            0.79,
            txt,
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax_21.transAxes,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="k"),
        )
        ax_21.legend(loc="upper left", markerscale=5)

    if save:
        dir_name = f"{PACKAGEDIR}/data/figures/tpf/ch{channel:02}"
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        fig_name = f"{dir_name}/dashboard_q{quarter:02}_ch{channel:02}_bkg{str(bkg)[0]}_augT_sgmT.pdf"
        print(fig_name)
        plt.savefig(fig_name, format="pdf", bbox_inches="tight", pad_inches=0.1)
    else:
        plt.show()


def plot_features(
    features,
    use_features=["Amplitude", "LinearTrend", "FluxPercentileRatioMid20"],
):
    feat_kp_pdc = features["feat_kp_pdc"]
    feat_kp_sap = features["feat_kp_sap"]
    feat_jm_sap = features["feat_jm_sap"]
    feat_jm_psfnv = features["feat_jm_psfnv"]
    feat_jm_psf = features["feat_jm_psf"]

    fig, ax = plt.subplots(2, 3, figsize=(19, 9))

    fig.suptitle("Feature Distribution for LCs", x=0.5, y=0.92)

    np.log10(feat_jm_sap["Amplitude"]).plot(
        kind="kde", ax=ax[0, 0], label="KBonus_SAP", color="tab:blue"
    )
    np.log10(feat_jm_psfnv["Amplitude"]).plot(
        kind="kde", ax=ax[0, 0], label="KBonus_PSFNV", color="tab:green"
    )
    np.log10(feat_jm_psf["Amplitude"]).plot(
        kind="kde", ax=ax[0, 0], label="KBonus_PSF", color="tab:orange"
    )
    # ax[0,0].set_xlim(0, 7)
    ax[0, 0].set_xlabel("Amplitude")
    ax[0, 0].legend(loc="upper right")

    (feat_jm_sap["Rcs"]).plot(
        kind="kde", ax=ax[0, 1], label="_nolegend_", color="tab:blue"
    )
    (feat_jm_psfnv["Rcs"]).plot(
        kind="kde", ax=ax[0, 1], label="_nolegend_", color="tab:green"
    )
    (feat_jm_psf["Rcs"]).plot(
        kind="kde", ax=ax[0, 1], label="_nolegend_", color="tab:orange"
    )
    ax[0, 1].axvline(0, c="tab:red", lw=1, label="Expected for symetric Flux dist")
    # ax[0,1].set_xlim(-)
    ax[0, 1].set_xlabel("Rcs")
    ax[0, 1].legend(loc="upper left")

    (feat_jm_sap["LinearTrend"]).plot(
        kind="kde", ax=ax[0, 2], label="KBonus_SAP", color="tab:blue"
    )
    (feat_jm_psfnv["LinearTrend"]).plot(
        kind="kde", ax=ax[0, 2], label="KBonus_PSFNV", color="tab:green"
    )
    (feat_jm_psf["LinearTrend"]).plot(
        kind="kde", ax=ax[0, 2], label="KBonus_PSF", color="tab:orange"
    )
    ax[0, 2].set_xlabel("LinearTrend")
    ax[0, 2].set_xlim(-0.5e-1, 0.5e-1)
    # ax[0,2].legend(loc="upper right")

    (feat_jm_sap["PercentDifferenceFluxPercentile"]).plot(
        kind="kde", ax=ax[1, 0], label="KBonus_SAP", color="tab:blue"
    )
    (feat_jm_psfnv["PercentDifferenceFluxPercentile"]).plot(
        kind="kde", ax=ax[1, 0], label="KBonus_PSFNV", color="tab:green"
    )
    (feat_jm_psf["PercentDifferenceFluxPercentile"]).plot(
        kind="kde", ax=ax[1, 0], label="KBonus_PSF", color="tab:orange"
    )
    ax[1, 0].set_xlabel(r"$(F_{95} - F_{5}) / median(F)$")
    ax[1, 0].set_xlim(-1, 1)
    # ax[1,0].legend(loc="upper right")

    (feat_jm_sap["FluxPercentileRatioMid20"]).plot(
        kind="kde", ax=ax[1, 1], label="_nolegend_", color="tab:blue"
    )
    (feat_jm_psfnv["FluxPercentileRatioMid20"]).plot(
        kind="kde", ax=ax[1, 1], label="_nolegend_", color="tab:green"
    )
    (feat_jm_psf["FluxPercentileRatioMid20"]).plot(
        kind="kde", ax=ax[1, 1], label="_nolegend_", color="tab:orange"
    )
    ax[1, 1].axvline(0.154, c="tab:red", lw=1, label="Expected for Flux ~ N(0,1)")
    # ax[1,1].set_xlim(-1.5, 0.5)
    ax[1, 1].set_xlabel(r"$(F_{60} - F_{40}) / (F_{95} - F_{5})$")
    ax[1, 1].legend(loc="upper right")

    (feat_jm_sap["FluxPercentileRatioMid80"]).plot(
        kind="kde", ax=ax[1, 2], label="_nolegend_", color="tab:blue"
    )
    (feat_jm_psfnv["FluxPercentileRatioMid80"]).plot(
        kind="kde", ax=ax[1, 2], label="_nolegend_", color="tab:green"
    )
    (feat_jm_psf["FluxPercentileRatioMid80"]).plot(
        kind="kde", ax=ax[1, 2], label="_nolegend_", color="tab:orange"
    )
    ax[1, 2].axvline(0.779, c="tab:red", lw=1, label="Expected for Flux ~ N(0,1)")
    ax[1, 2].set_xlabel(r"$(F_{90} - F_{10}) / (F_{95} - F_{5})$")
    ax[1, 2].legend(loc="upper left")

    if isinstance(feat_kp_pdc, pd.DataFrame):
        np.log10(feat_kp_pdc["Amplitude"]).plot(
            kind="kde", ax=ax[0, 0], label="KepPipe_PDC", color="k"
        )
        np.log10(feat_kp_sap["Amplitude"]).plot(
            kind="kde", ax=ax[0, 0], label="KepPipe_SAP", color="k", ls="--"
        )
        (feat_kp_pdc["Rcs"]).plot(
            kind="kde", ax=ax[0, 1], label="_nolegend_", color="k"
        )
        (feat_kp_sap["Rcs"]).plot(
            kind="kde", ax=ax[0, 1], label="_nolegend_", color="k", ls="--"
        )
        (feat_kp_pdc["LinearTrend"]).plot(
            kind="kde", ax=ax[0, 2], label="KepPipe_PDC", color="k"
        )
        (feat_kp_sap["LinearTrend"]).plot(
            kind="kde", ax=ax[0, 2], label="KepPipe_SAP", color="k", ls="--"
        )
        (feat_kp_pdc["PercentDifferenceFluxPercentile"]).plot(
            kind="kde", ax=ax[1, 0], label="KepPipe_PDC", color="k"
        )
        (feat_kp_sap["PercentDifferenceFluxPercentile"]).plot(
            kind="kde", ax=ax[1, 0], label="KepPipe_SAP", color="k", ls="--"
        )
        (feat_kp_pdc["FluxPercentileRatioMid20"]).plot(
            kind="kde", ax=ax[1, 1], label="_nolegend_", color="k"
        )
        (feat_kp_sap["FluxPercentileRatioMid20"]).plot(
            kind="kde", ax=ax[1, 1], label="_nolegend_", color="k", ls="--"
        )
        (feat_kp_pdc["FluxPercentileRatioMid80"]).plot(
            kind="kde", ax=ax[1, 2], label="_nolegend_", color="k"
        )
        (feat_kp_sap["FluxPercentileRatioMid80"]).plot(
            kind="kde", ax=ax[1, 2], label="_nolegend_", color="k", ls="--"
        )

        ax[0, 2].set_xlim(-1e-3, 1e-3)
        ax[0, 2].set_ylim(-0.1, 5e3)
        # ax[0, 2].set_xlim(-0.5e-1, 0.5e-1)
        ax[1, 0].set_xlim(-0.5, 0.5)
        # ax[1, 2].set_xlim(0.4, 1.2)

    plt.show()

    if isinstance(feat_kp_pdc, pd.DataFrame):
        concat = pd.concat(
            [
                feat_jm_sap.query("Amplitude < .1").assign(dataset="SAP"),
                feat_jm_psf.query("Amplitude < .1").assign(dataset="PSF"),
                feat_kp_pdc.query("Amplitude < .1").assign(dataset="PDC"),
            ]
        )
        palette = {"PDC": "k", "SAP": "tab:blue", "PSF": "tab:orange"}
    else:
        concat = pd.concat(
            [
                feat_jm_sap.query("Amplitude < .1").assign(dataset="SAP"),
                feat_jm_psf.query("Amplitude < .1").assign(dataset="PSF"),
            ]
        )
        palette = {"SAP": "tab:blue", "PSF": "tab:orange"}

    sb.pairplot(
        concat,
        hue="dataset",
        palette=palette,
        vars=use_features,
        plot_kws={"s": 4, "alpha": 0.5},
    )
    plt.show()
