import os, sys, re
import socket
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from glob import glob
import lightkurve as lk
from astropy.io import fits
import fitsio
import matplotlib.pyplot as plt

sys.path.append(f"{os.path.dirname(os.getcwd())}/kepler_workflow/")
from paths import *

import astropy.units as u

import warnings

warnings.filterwarnings("ignore", category=lk.LightkurveWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

dmcs = {}
typedir = {
    int: "J",
    str: "A",
    float: "D",
    bool: "L",
    np.int32: "J",
    np.int64: "K",
    np.float32: "E",
    np.float64: "D",
}
qd_map = {
    0: 2009131105131,
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
keep_col = [
    "time",
    "cadenceno",
    "quarter",
    "flux",
    "flux_err",
    "sap_flux",
    "sap_flux_err",
    "psf_flat_flux",
    "psf_flat_flux_err",
    "sap_quality",
    "flatten_mask",
]
per_quarter = [
    "SEASON",
    "MODULE",
    "OUTPUT",
    "QUARTER",
    "TPF_ROW",
    "TPF_COL",
    "SAP",
    "ROW",
    "COLUMN",
    "FLFRCSAP",
    "CROWDSAP",
    "NPIXSAP",
    "PSFFRAC",
    "PERTRATI",
    "PERTSTD",
    "ITERNEG",
]
psf_frac = 0.4


def fancy_flatten(
    lc,
    spline_dt=1.5,
    comp=["spline"],
    force=False,
    do_break_mask=False,
    name="norm_",
    period=None,
    t0=None,
    copy=True,
    poly_deg=3,
    duration=None,
    plot=False,
    correction="sub",
):
    # creating Design matrix
    quarter = lc.quarter
    channel = lc.channel
    module = lc.module
    output = lc.output
    year = str(qd_map[quarter])
    dmc_name = f"{quarter:02}_{channel:02}_{'-'.join(comp)}"
    lc = lc.remove_nans()
    # lc = lc[lc.flux > 0]
    # lc = lc[lc.flux_err > 0]
    # lc = lc[(lc.time.bkjd > 0) & (lc.time.bkjd < 1e5)]
    if len(lc) == 0:
        print("LC has all-nan values in PSF column")
        if copy:
            return lk.LightCurve(
                time=lc.time,
                flux=u.Quantity(lc.flux, unit=lc.flux.unit),
                flux_err=u.Quantity(lc.flux_err, unit=lc.flux.unit),
                quality=lc.sap_quality,
            )
        else:
            return lc

    times = lc.time.bkjd
    if dmc_name in dmcs.keys() and not force:
        dmc, breaks, _breaks = dmcs[dmc_name]
    else:
        bkg_file = (
            f"{ARCHIVE_PATH}/data/kepler/bkg/"
            f"{year[:4]}/kplr{module:02}{output}-{year}_bkg.fits.gz"
        )
        if not os.path.isfile(bkg_file):
            print("BKG file doesn't exist:")
            print(bkg_file)
            sys.exit(1)

        poscorr = fitsio.read(
            bkg_file,
            columns=["CADENCENO", "POS_CORR1", "POS_CORR2"],
            ext=1,
        )
        cad, mpc1, mpc2 = (
            poscorr["CADENCENO"],
            poscorr["POS_CORR1"][:, 2000],
            poscorr["POS_CORR2"][:, 2000],
        )
        mask = np.in1d(cad, lc.cadenceno.value)
        cad, mpc1, mpc2 = cad[mask], mpc1[mask], mpc2[mask]

        splits = np.where(np.diff(times) > 0.3)[0] + 1

        grads1 = np.gradient(mpc1, times)
        grads2 = np.gradient(mpc2, times)

        splits1 = np.where(grads1 > 7 * grads1.std())[0]
        splits2 = np.where(grads2 > 7 * grads2.std())[0]
        # merging breaks
        # _breaks = splits
        _breaks = np.unique(np.concatenate([splits, splits1[0::2], splits2[0::2]]))
        _breaks = np.delete(_breaks, np.where((_breaks[1:] - _breaks[:-1]) < 5)[0])
        breaks = list(_breaks[(_breaks > 5) & (_breaks < len(times) - 5)])

        # collection of DMs
        dmc = []
        if "offset" in comp:
            # offset matrix
            offset_dm = lk.DesignMatrix(
                np.ones_like(times), name="offset"
            )  # .split(breaks)
            offset_dm.prior_mu = np.ones(1)
            offset_dm.prior_sigma = np.ones(1) * 1e7
            dmc.append(offset_dm)
        if "cbv" in comp:
            # DM with CBVs, first 4 only
            cbvs = lk.correctors.download_kepler_cbvs(
                mission="Kepler", quarter=quarter, channel=channel
            )
            basis = 2
            cbv_dm = cbvs[mask].to_designmatrix()
            cbv_dm = lk.DesignMatrix(cbv_dm.X[:, :basis], name="cbv").split(breaks)
            cbv_dm.prior_mu = np.zeros(basis * (len(breaks) + 1))
            cbv_dm.prior_sigma = np.ones(basis * (len(breaks) + 1)) * 1e4
            dmc.append(cbv_dm)
        if "spline" in comp:
            # spline DM
            n_knots = int((times[-1] - times[0]) / spline_dt)
            spline_dm = lk.designmatrix.create_spline_matrix(
                times,
                n_knots=n_knots,
                include_intercept=True,
                name="spline",
            ).split(breaks)
            spline_dm = lk.DesignMatrix(
                spline_dm.X[:, spline_dm.X.sum(axis=0) != 0], name="spline"
            )
            dmc.append(spline_dm)
        if "time-poly" in comp:
            # time polys
            time_norm = (times - np.nanmean(times)) / np.std(times)
            poly_dm = lk.DesignMatrix(
                np.vstack([time_norm ** idx for idx in range(poly_deg + 1)]).T,
                name="time_poly",
            )  # .split(breaks)
            # poly_dm = lk.DesignMatrix(
            #     poly_dm.X[:, poly_dm.X.sum(axis=0) != 0], name="time_poly"
            # )
            poly_dm.prior_mu = np.ones((poly_deg + 1))
            poly_dm.prior_sigma = np.ones((poly_deg + 1)) * 1e7
            dmc.append(poly_dm)
        dmc = lk.DesignMatrixCollection(dmc)
        dmcs[dmc_name] = [dmc, breaks, _breaks]

    cadence_mask = np.ones(len(times), dtype=bool)
    if do_break_mask and len(breaks) > 0:
        break_mask = np.array(_breaks)
        break_mask = np.unique(
            np.concatenate(
                [
                    break_mask,
                    break_mask + 1,
                    break_mask - 1,
                    break_mask + 2,
                    break_mask - 2,
                    break_mask + 3,
                    break_mask - 3,
                ]
            )
        )
        break_mask = break_mask[(break_mask >= 0) & (break_mask < len(times))]
        cadence_mask[break_mask] = False
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

    rc = lk.RegressionCorrector(lc)
    try:
        clc = rc.correct(dmc, sigma=3, cadence_mask=cadence_mask, propagate_errors=True)
    except (np.linalg.LinAlgError, IndexError):
        try:
            clc = rc.correct(dmc, sigma=3, cadence_mask=None, propagate_errors=True)
        except (np.linalg.LinAlgError, IndexError):
            print(
                f"Warning: flattening failed with Singular matrix for "
                f"source {lc.TARGETID} quarter {quarter}."
            )
            if "spline" in comp:
                print("Using .flatten()")
                clc = lc.flatten(window_length=101, polyorder=2)
                clc.flux *= np.nanmedian(lc.flux)
                clc.flux_err = lc.flux_err
            else:
                print("Using original LC")
                clc = lc.copy()

    if correction == "div":
        clc = lc.copy()
        clc.flux = lc.flux / (np.nanmedian(lc.flux) + rc.model_lc.flux)
        clc.flux *= np.nanmedian(lc.flux)
        clc.flux_err = np.sqrt(lc.flux_err ** 2 + rc.model_lc.flux_err ** 2)

    quality_mask = np.zeros_like(lc.sap_quality)
    if do_break_mask and len(breaks) > 0:
        quality_mask[break_mask] = 1
    if plot:
        fig, ax = plt.subplots(1, len(comp), figsize=(len(comp) * 3, 3))
        if len(comp) == 1:
            dmc[0].plot(ax=ax)
        else:
            for k in range(len(comp)):
                dmc[k].plot(ax=ax[k])
        plt.show()
        axs = rc.diagnose()
        axs[0].vlines(
            lc.time.bkjd[breaks],
            ymin=axs[0].get_ylim()[0],
            ymax=axs[0].get_ylim()[1],
            color="tab:red",
            lw=1,
        )
        plt.show()

    if copy:
        return lk.LightCurve(
            time=clc.time,
            flux=u.Quantity(clc.flux, unit=clc.flux.unit),
            flux_err=u.Quantity(clc.flux_err, unit=clc.flux.unit),
            quality=quality_mask,
        )
    else:
        lc[f"{name}flux"] = clc["flux"]
        lc[f"{name}flux_err"] = clc["flux_err"]
        lc[f"{name}quality"] = quality_mask
        return lc


def get_lc(name, force=False, skip_qs=[]):

    if not force:
        stitched_file = glob(
            f"{KBONUS_LCS_PATH}/kepler/{name[:4]}_s/{name}/"
            f"hlsp_kbonus-bkg_kepler_kepler_*-{name}*_lc.fits"
        )
        if len(stitched_file) > 0:
            return None, None

    lc_files = sorted(
        glob(
            f"{KBONUS_LCS_PATH}/kepler/{name[:4]}/{name}/"
            f"hlsp_kbonus-bkg_kepler_kepler_*-{name}-q*_lc.fits"
        )
    )
    if len(lc_files) == 0:
        return None, None
    qs = np.array([x.split("/")[-1].split("-")[-1][1:3] for x in lc_files], dtype=int)

    # check for crrelated LCs to skip
    lc = lk.LightCurveCollection(
        [lk.KeplerLightCurve.read(x) for x, q in zip(lc_files, qs) if q not in skip_qs]
    )
    if len(lc) == 0:
        return None, None

    return lc, lc_files


def process_and_stitch(lc, do_flat=True, do_align=True):
    if not do_flat and not do_align:
        raise ValueError("Both flat and align are set to False")
    lc_flat = []
    lc_psf_align, lc_sap_align = [], []
    mean_flux_psf = []
    mean_flux_sap = []
    q_mask = np.zeros(len(lc), dtype=bool)
    for i, x in enumerate(lc):
        # check the times are ok
        # if np.abs(x.time[1:].bkjd - x.time[:-1].bkjd).max() > 20:
        #     print("WARNING: Times are wrong for ", x.meta["FILENAME"])
        #     continue
        if do_flat:
            if x.meta["PSFFRAC"] <= psf_frac:
                flat = lk.LightCurve(
                    time=x.time,
                    flux=x.flux * np.nan,
                    flux_err=x.flux_err * np.nan,
                    quality=x.quality,
                )
            else:
                flat = fancy_flatten(
                    x,
                    correction="sub",
                    spline_dt=1.5,
                    plot=False,
                    do_break_mask=True,
                    copy=True,
                    force=False,
                    comp=["spline"],
                    name="",
                )
            mean_flux_psf.extend(flat.flux[flat.quality != 1].value)
            lc_flat.append(flat)
        if do_align:
            if x.meta["PSFFRAC"] <= psf_frac:
                align_psf = lk.LightCurve(
                    time=x.time,
                    flux=x.flux * np.nan,
                    flux_err=x.flux_err * np.nan,
                    quality=x.quality,
                )
            else:
                align_psf = fancy_flatten(
                    x,
                    correction="sub",
                    spline_dt=1.5,
                    plot=False,
                    do_break_mask=True,
                    copy=True,
                    force=False,
                    comp=["time-poly"],
                    name="",
                )
            x["flux"] = x["sap_flux"]
            x["flux_err"] = x["sap_flux_err"]
            align_sap = fancy_flatten(
                x,
                correction="sub",
                spline_dt=1.5,
                plot=False,
                do_break_mask=True,
                copy=True,
                force=False,
                comp=["time-poly"],
                name="",
            )
            lc_psf_align.append(align_psf)
            lc_sap_align.append(align_sap)
            mean_flux_sap.extend(x.sap_flux.value)
            q_mask[i] = True

    mean_flux_psf = np.nanmean(mean_flux_psf)
    mean_flux_sap = np.nanmean(mean_flux_sap)

    lc_stitch = []
    for fl, pa, sa, x in zip(lc_flat, lc_psf_align, lc_sap_align, lc):
        fl["flux"] = (fl["flux"] / np.nanmean(fl["flux"].value)) * mean_flux_psf
        pa["flux"] = (pa["flux"] / np.nanmean(pa["flux"].value)) * mean_flux_psf
        sa["flux"] = (sa["flux"] / np.nanmean(sa["flux"].value)) * mean_flux_sap
        _lc_stitch = pa.copy()
        _lc_stitch["sap_flux"] = sa["flux"]
        _lc_stitch["sap_flux_err"] = sa["flux_err"]
        _lc_stitch["psf_flat_flux"] = fl["flux"]
        _lc_stitch["psf_flat_flux_err"] = fl["flux_err"]
        _lc_stitch["sap_quality"] = x["sap_quality"]
        _lc_stitch["flatten_mask"] = _lc_stitch["quality"]
        _lc_stitch["cadenceno"] = x["cadenceno"]
        _lc_stitch["quarter"] = x.quarter
        del _lc_stitch["quality"]
        lc_stitch.append(_lc_stitch)

    lc_stitch = lk.LightCurveCollection(lc_stitch).stitch(corrector_func=lambda x: x)

    return lc_stitch


def make_fits(name, lc_stitch, lc_files=None, quarter_skip=[]):
    if lc_files is None:
        lc_files = sorted(
            glob(
                f"{KBONUS_LCS_PATH}/kepler/{name[:4]}/{name}/"
                f"hlsp_kbonus-bkg_kepler_kepler_*-{name}-q*_lc.fits"
            )
        )

    hduls = []
    quarters_detect = [0] * 18
    for k, f in enumerate(lc_files):
        aux = fits.open(f)
        if aux[0].header["QUARTER"] in quarter_skip:
            print(f"WARNING: Skiping correlated LC for Q {aux[0].header['QUARTER']}")
            continue
        if k == 0:
            primhdu = aux[0].copy()
            for kw in per_quarter:
                aux[1].header.set(
                    kw,
                    value=primhdu.header[kw],
                    comment=primhdu.header.comments[kw],
                    before="EXTNAME",
                )
                primhdu.header.remove(kw)
            hduls.append(primhdu)
        for kw in per_quarter:
            aux[1].header.set(
                kw,
                value=aux[0].header[kw],
                comment=aux[0].header.comments[kw],
                before="EXTNAME",
            )
        aux[1].header["EXTNAME"] = f"LIGHTCURVE_Q{aux[0].header['QUARTER']}"
        try:
            aux[2].header["EXTNAME"] = f"APERTURE_Q{aux[0].header['QUARTER']}"
        except:
            print(
                f"WARNING: no aperture extension in Q {aux[0].header['QUARTER']}"
                f"for file {f}"
            )
        quarters_detect[aux[0].header["QUARTER"]] = (
            1 if aux[0].header["PSFFRAC"] > psf_frac else 2
        )
        hduls.extend(aux[1:])

    # make stitch lc hdu
    coldefs = []
    for key in keep_col:
        if key == "time":
            arr = lc_stitch[key].bkjd
        else:
            arr = lc_stitch[key]
        arr_unit = ""
        arr_type = typedir[arr.dtype.type]
        if "flux" in key:
            arr_unit = "e-/s"
        elif "time" == key:
            arr_unit = "jd"
        coldefs.append(
            fits.Column(
                name=key.upper(),
                array=arr,
                unit=arr_unit,
                format=arr_type,
            )
        )

    hdu_stitch = fits.BinTableHDU.from_columns(coldefs)
    hdu_stitch.header["EXTNAME"] = "LIGHTCURVE_STITCHED"
    hduls.insert(1, hdu_stitch)
    hduls = fits.HDUList(hduls)

    hduls[0].header.set(
        "QDETECT",
        value=''.join([str(x) for x in quarters_detect]),
        comment="Detection flag per quarter",
    )
    quarter = hduls[2].header["QUARTER"]
    outname = lc_files[0].replace(f"/{name[:4]}/", f"/{name[:4]}_s/")
    outname = outname.replace(f"-q{quarter:02}_", "_")
    outname = outname.replace(f"v1.1.1", "v1.0")

    if not os.path.isdir(f"{KBONUS_LCS_PATH}/kepler/{name[:4]}_s/"):
        os.makedirs(f"{KBONUS_LCS_PATH}/kepler/{name[:4]}_s/")
    if not os.path.isdir(f"{KBONUS_LCS_PATH}/kepler/{name[:4]}_s/{name}"):
        os.makedirs(f"{KBONUS_LCS_PATH}/kepler/{name[:4]}_s/{name}")

    hduls.writeto(outname, overwrite=True, checksum=True)
    return hduls


def get_bkg_file_names(lc):
    bkg_files = []
    for x in lc:
        bkfname = (
            f"{ARCHIVE_PATH}/data/kepler/bkg/"
            f"{str(qd_map[x.quarter])[:4]}/"
            f"kplr{x.module:02}{x.output}-{str(qd_map[x.quarter])}_bkg.fits.gz"
        )
        if not os.path.isfile(bkfname):
            bkg_files.append(
                f"{str(qd_map[x.quarter])[:4]}"
                f"/kplr{x.module:02}{x.output}-{str(qd_map[x.quarter])}_bkg.fits.gz"
            )

    return np.array(bkg_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stitch FITS files")
    parser.add_argument(
        "--dir",
        dest="dir",
        type=str,
        default="0072",
        help="Directory 4 digit string",
    )
    args = parser.parse_args()
    source_names = np.array(os.listdir(f"{KBONUS_LCS_PATH}/kepler/{args.dir}")).ravel()

    correlated = pd.read_csv(
        f"{PACKAGEDIR}/data/catalogs/tpf/correlated_lcs_to_remove.csv", index_col=0
    )

    fails = []
    bkg_files = []
    for k, fname in tqdm(
        enumerate(source_names), total=len(source_names), desc=f"Dir {args.dir}"
    ):
        skip_qs = correlated.query(f"fname == '{fname}'").quarter.values
        lc, lc_files = get_lc(fname, force=False, skip_qs=skip_qs)
        if lc is None:
            continue
        try:
            lc_flat = process_and_stitch(lc, do_flat=True, do_align=True)
        except ValueError:
            try:
                lc_flat = process_and_stitch(lc, do_flat=True, do_align=True)
            except:
                fails.append(fname)
                continue
        try:
            _ = make_fits(fname, lc_flat, lc_files=lc_files, quarter_skip=skip_qs)
        except KeyError:
            fails.append(fname)
            print(f"{fname} failed with Keyword 'QUARTER' not found.")
            continue

    print(fails)
    np.savetxt(
        f"{PACKAGEDIR}/data/support/failed_stitch_{args.dir}.txt",
        np.array([f"{x[:4]}/{x}" for x in fails]),
        fmt="%s",
    )
    print("Done!")


'''
Failed:

module 13
['009462617']
['2126348255678300928', '2126621656115682048']

m-starts
002974627 003101896
'''
