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

if socket.gethostname() in ["NASAs-MacBook-Pro.local", "NASAs-MacBook-Pro-2.local"]:
    ssd_kbonus = "/Volumes/jorge-marpa-personal/work/kbonus"
    ssd_kepler = "/Volumes/jorge-marpa/Work/BAERI/data/kepler"
else:
    ssd_kbonus = "/Volumes/jorge-marpa-personal/work/kbonus"
    ssd_kepler = "/Volumes/jorge-marpa/Work/BAERI/data/kepler"

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
    if len(lc.remove_nans()) == 0:
        # print("LC has all-nan values in PSF column")
        return lc
    else:
        lc = lc.remove_nans()
    times = lc.time.bkjd
    if dmc_name in dmcs.keys() and not force:
        dmc, breaks, _breaks = dmcs[dmc_name]
    else:
        poscorr = fitsio.read(
            f"{ARCHIVE_PATH}/data/kepler/bkg/"
            f"{year[:4]}/kplr{module}{output}-{year}_bkg.fits.gz",
            columns=["CADENCENO", "POS_CORR1", "POS_CORR2"],
            ext=1,
        )
        cad, mpc1, mpc2 = (
            poscorr["CADENCENO"],
            poscorr["POS_CORR1"][:, 4184],
            poscorr["POS_CORR2"][:, 4184],
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
    # print(f"Masking total cadences {(~cadence_mask).sum()} {(cadence_mask).sum()}")
    rc = lk.RegressionCorrector(lc)
    try:
        clc = rc.correct(dmc, sigma=3, cadence_mask=cadence_mask, propagate_errors=True)
    except np.linalg.LinAlgError:
        try:
            clc = rc.correct(dmc, sigma=3, cadence_mask=None, propagate_errors=True)
        except np.linalg.LinAlgError:
            print(
                f"Warning: flattening failed with Singular matrix for "
                f"source {lc.TARGETID} quarter {quarter}. Skipping"
            )
            return None

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


def get_lc(name, force=False):

    if not force:
        if (
            len(
                glob(
                    f"{LCS_PATH}/kepler/mstars/{name[:4]}/{name}/"
                    f"hlsp_kbonus-bkg_kepler_kepler_*-{name}_kepler_v1.1.1_lc.fits"
                )
            )
            != 0
        ):
            return None
    lc_files = sorted(
        glob(
            f"{LCS_PATH}/kepler/mstars/{name[:4]}/{name}/"
            f"hlsp_kbonus-bkg_kepler_kepler_*-{name}-q*_lc.fits"
        )
    )
    if len(lc_files) == 0:
        print(name)
        return None

    lc = lk.LightCurveCollection([lk.KeplerLightCurve.read(x) for x in lc_files])

    # lc = lk.LightCurveCollection([x for x in lc if np.isfinite(x.flux).all()])
    # if len(lc) == 0:
    #     print(name)
    #     return None

    return lc


def process_and_stitch(lc, do_flat=True, do_align=True):
    if not do_flat and not do_align:
        raise ValueError("Both flat and align are set to False")
    lc_flat = []
    lc_psf_align, lc_sap_align = [], []
    mean_flux_psf = []
    mean_flux_sap = []
    quarter_mask = np.zeros(len(lc), dtype=bool)
    for i, x in enumerate(lc):
        if do_flat:
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
            if flat is None:
                continue
            mean_flux_psf.extend(flat.flux[flat.quality != 1].value)
            lc_flat.append(flat)
        if do_align:
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
        quarter_mask[i] = True

    mean_flux_psf = np.nanmean(mean_flux_psf)
    mean_flux_sap = np.nanmean(mean_flux_sap)

    lc_stitch = []
    for fl, pa, sa, x in zip(lc_flat, lc_psf_align, lc_sap_align, lc[quarter_mask]):
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

    return lc_stitch, quarter_mask


def make_fits(name, lc_stitch, quarter_mask=None):
    lc_files = sorted(
        glob(
            f"{LCS_PATH}/kepler/mstars/{name[:4]}/{name}/"
            f"hlsp_kbonus-bkg_kepler_kepler_*-{name}-q*_lc.fits"
        )
    )
    if quarter_mask is None:
        quarter_mask = np.ones(len(lc_files), dtype=bool)

    hduls = []
    for k, f in enumerate(lc_files):
        if not quarter_mask[k]:
            continue
        aux = fits.open(f)
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

    quarter = hduls[2].header["QUARTER"]
    outname = lc_files[0].replace(f"-q{quarter:02}_", "_")

    # print(hduls.info())
    hduls.writeto(outname, overwrite=True, checksum=True)
    return


def get_bkg_file_names(lc):
    bkg_files = [
        f"{str(qd_map[x.quarter])[:4]}/kplr{x.module}{x.output}-{str(qd_map[x.quarter])}_bkg.fits.gz"
        for x in lc
    ]
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
    source_names = np.array(os.listdir(f"{LCS_PATH}/kepler/mstars/{args.dir}")).ravel()

    fails = []
    for k, fname in tqdm(enumerate(source_names), total=len(source_names)):
        lc = get_lc(fname, force=True)
        if lc is None:
            continue
        # bkg_files.extend(get_bkg_file_names(lc))
        lc_flat, quarter_mask = process_and_stitch(lc, do_flat=True, do_align=True)
        try:
            make_fits(fname, lc_flat, quarter_mask=quarter_mask)
        except KeyError:
            fails.append(fname)
            print(f"{fname} failes with Keyword 'QUARTER' not found.")
            continue

    print(fails)
    print("Done!")


'''
Failed:
['009462617']
['2126348255678300928', '2126621656115682048']
'''
