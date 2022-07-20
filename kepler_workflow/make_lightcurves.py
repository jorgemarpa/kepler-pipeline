import os
import sys
import argparse
import socket
import yaml
import tarfile
import tempfile
import logging
from glob import glob
from datetime import datetime
from time import sleep
import numpy as np
import pandas as pd
import psfmachine as pm
import lightkurve as lk
from scipy import stats
from tqdm import tqdm

# from memory_profiler import profile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
from astropy.io import fits
import fitsio
from psfmachine.utils import _make_A_polar, bspline_smooth
from psfmachine.aperture import aperture_mask_to_2d

# from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_3d
import astropy.units as u
from astroquery.vizier import Vizier

from paths import ARCHIVE_PATH, OUTPUT_PATH, LCS_PATH, PACKAGEDIR

import warnings
from scipy import sparse

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)

logg = logging.getLogger("Make LCs")
lc_version = "1.1.1"

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


def print_dict(dictionary):
    for k in sorted(dictionary.keys()):
        logg.info(f"{k:<22}: {dictionary[k]}")


# @profile
def get_KICs(catalog):
    """
    Query KIC (<2") and return the Kepler IDs for result sources.
    """
    gaia = SkyCoord(ra=catalog.ra, dec=catalog.dec, frame="icrs", unit=(u.deg, u.deg))
    v = Vizier(catalog=["V/133/kic"])
    kic = v.query_region(gaia, radius=2 * u.arcsec, catalog=["V/133/kic"])["V/133/kic"]
    kicc = SkyCoord(
        ra=kic["RAJ2000"], dec=kic["DEJ2000"], frame="icrs", unit=(u.deg, u.deg)
    )
    midx, mdist = match_coordinates_3d(gaia, kicc, nthneighbor=1)[:2]
    catalog.loc[:, "kic"] = 0
    catalog.loc[mdist.arcsec < 2, "kic"] = kic[midx[mdist.arcsec < 2]]["KIC"].data.data
    catalog["kic"] = catalog["kic"].astype(int)
    catalog.loc[:, "kepmag"] = np.nan
    catalog.loc[mdist.arcsec < 2, "kepmag"] = kic[midx[mdist.arcsec < 2]][
        "kepmag"
    ].data.data
    return catalog


# @profile
def get_file_list(quarter, channel, batch_number=-1, tar_tpfs=True):

    lookup_table = pd.read_csv(
        "%s/data/support/kepler_tpf_map_q%02i%s_new.csv"
        % (PACKAGEDIR, quarter, "_tar" if tar_tpfs else ""),
        index_col=0,
    )
    files_in = lookup_table.query(f"channel == {channel} and quarter =={quarter}")
    if files_in.shape[0] == 0:
        raise IndexError("Channel does not contain TPFs.")
    if batch_number > 0:
        if batch_number not in files_in.batch.unique():
            raise ValueError("Asked batch is not in file")
        files_in = files_in.query(f"batch == {batch_number}")
        if files_in.shape[0] == 0:
            raise IndexError("Batch does not contain TPFs.")
        return files_in.file_name.tolist()
    else:
        return files_in.file_name.tolist()


# @profile
def make_hdul(data, lc_meta, extra_meta, aperture_mask=None):
    meta = {
        "ORIGIN": (lc_meta["ORIGIN"], "Program used for photometry"),
        "VERSION": (pm.__version__, "Version of ORIGIN"),
        "APERTURE": (lc_meta["APERTURE"], "Type of photometry in file"),
        # telescope info
        "MISSION": (lc_meta["MISSION"], "Mission"),
        "TELESCOP": (extra_meta["TELESCOP"], "Telescope name"),
        "INSTRUME": (extra_meta["INSTRUME"], "Telescope instrument"),
        "OBSMODE": (extra_meta["OBSMODE"], "Observation mode"),
        "SEASON": (extra_meta["SEASON"], "Observation season"),
        "CHANNEL": (lc_meta["CHANNEL"], "CCD channel"),
        "MODULE": (lc_meta["MODULE"], "CCD module"),
        "OUTPUT": (extra_meta["OUTPUT"], "CCD module output"),
        "QUARTER": (lc_meta["QUARTER"], "Observation quarter"),
        # "CAMPAIGN": lc_meta["CAMPAIGN"],
        # objct info
        "LABEL": (
            f"KIC {extra_meta['KEPLERID']}"
            if extra_meta["KEPLERID"] != 0
            else extra_meta["GAIA_DES"],
            "Object label",
        ),
        "TARGETID": (lc_meta["TARGETID"], "Kepler target identifier"),
        "RA_OBJ": (lc_meta["RA"], "Right ascension [deg]"),
        "DEC_OBJ": (lc_meta["DEC"], "Declination [deg]"),
        "EQUINOX": (extra_meta["EQUINOX"], "Coordinate equinox"),
        # KIC info
        "KEPLERID": (
            extra_meta["KEPLERID"] if extra_meta["KEPLERID"] != 0 else "",
            "Kepler identifier",
        ),
        "KEPMAG": (
            extra_meta["KEPMAG"] if np.isfinite(extra_meta["KEPMAG"]) else "",
            "Kepler magnitude [mag]",
        ),
        "TPFORG": (extra_meta["TPFORG"], "TPF id of object origin"),
        # gaia catalog info
        "GAIAID": (extra_meta["GAIA_DES"], "Gaia identifier"),
        "PMRA": (
            lc_meta["PMRA"] if np.isfinite(lc_meta["PMRA"]) else "",
            "Gaia RA proper motion [mas/yr]",
        ),
        "PMDEC": (
            lc_meta["PMDEC"] if np.isfinite(lc_meta["PMDEC"]) else "",
            "Gaia Dec proper motion [mas/yr]",
        ),
        "PARALLAX": (
            lc_meta["PARALLAX"] if np.isfinite(lc_meta["PARALLAX"]) else "",
            "Gaia parallax [mas]",
        ),
        "GMAG": (
            lc_meta["GMAG"] if np.isfinite(lc_meta["GMAG"]) else "",
            "Gaia G magnitude [mag]",
        ),
        "RPMAG": (
            lc_meta["RPMAG"] if np.isfinite(lc_meta["RPMAG"]) else "",
            "Gaia RP magnitude [mag]",
        ),
        "BPMAG": (
            lc_meta["BPMAG"] if np.isfinite(lc_meta["BPMAG"]) else "",
            "Gaia BP magnitude [mag]",
        ),
        # extraction info
        "TPF_ROW": (lc_meta["ROW"], "Origin pixel row in TPF"),
        "TPF_COL": (lc_meta["COLUMN"], "Origin pixel column in TPF"),
        "SAP": (
            lc_meta["SAP"],
            "SAP mode used in psmachine",
        ),
        "ROW": (
            np.nanmean(data["centroid_row"]).value,
            "Pixel row mean value of object SAP centroid ",
        ),
        "COLUMN": (
            np.nanmean(data["centroid_col"]).value,
            "Pixel column mean value of object SAP centroid ",
        ),
        "FLFRCSAP": (
            lc_meta["FLFRCSAP"] if np.isfinite(lc_meta["FLFRCSAP"]) else "",
            "Flux completeness metric for aperture",
        ),
        "CROWDSAP": (
            lc_meta["CROWDSAP"] if np.isfinite(lc_meta["CROWDSAP"]) else "",
            "Flux crowding metric for aperture",
        ),
        "NPIXSAP": (extra_meta["PIXINAP"], "Number of pixels in the aperture"),
        "PSFFRAC": (extra_meta["PSFFRAC"], "PSF fraction in data"),
        "PERTRATI": (extra_meta["PERRATIO"], "Ratio of perturbed and mean shape model"),
        "PERTSTD": (extra_meta["PERSTD"], "Standard deviation of perturbed model"),
        "ITERNEG": (
            extra_meta["ITERNEG"],
            "If object has negative psf_nova flux due to iter",
        ),
    }
    lc_dct = {
        "cadenceno": data["cadenceno"],
        "time": data["time"] - 2454833,
        "flux": data["flux"],
        "flux_err": data["flux_err"],
        "sap_flux": data["sap_flux"],
        "sap_flux_err": data["sap_flux_err"],
        "centroid_column": data["centroid_col"].value,
        "centroid_row": data["centroid_row"].value,
        "sap_quality": data["quality"],
        "sap_bkg": data["sap_bkg"],
        "red_chi2": data["red_chi2"],
    }
    if "psf_flux_NVA" in data.keys():
        lc_dct["psf_flux_nova"] = data["psf_flux_NVA"]
        lc_dct["psf_flux_err_nova"] = data["psf_flux_err_NVA"]

    lc_dict_ = lc_dct.copy()

    lc_ = lk.LightCurve(lc_dct, meta=meta)
    del lc_dct["time"], lc_dct["flux_err"], lc_dct["sap_quality"]
    hdul = lc_.to_fits(**lc_dct, **lc_.meta)

    # make Lightcurve header
    coldefs = []
    for key, data in lc_dict_.items():
        arr = data
        arr_unit = ""
        arr_type = typedir[data.dtype.type]
        if "flux" in key:
            arr_unit = "e-/s"
        elif "time" == key:
            arr_unit = "jd"
        elif "centroid" in key:
            arr_unit = "pix"
        coldefs.append(
            fits.Column(
                name=key.upper(),
                array=arr,
                unit=arr_unit,
                format=arr_type,
            )
        )
    hdul[1] = fits.BinTableHDU.from_columns(coldefs)
    hdul[1].header["EXTNAME"] = "LIGHTCURVE"

    for key, val in meta.items():
        hdul[0].header.set(key, val[0], val[1])

    if aperture_mask is not None:
        ap_hdu = fits.ImageHDU(aperture_mask.astype(np.uint8))
        ap_hdu.header["EXTNAME"] = "APERTURE"
        return fits.HDUList([hdul[0], hdul[1], ap_hdu])
    else:
        return hdul


# @profile
def get_tpfs(fname_list, tar_tpfs=True):
    if not tar_tpfs:
        return lk.collections.TargetPixelFileCollection(
            [
                lk.KeplerTargetPixelFile(
                    # f"{ARCHIVE_PATH}/data/kepler/tpf/{f}", quality_bitmask="none"
                    f,
                    quality_bitmask="none",
                )
                for f in fname_list
            ]
        )
    else:
        tpfs = []
        with tempfile.TemporaryDirectory(prefix="temp_fits") as tmpdir:
            for k, fname in enumerate(fname_list):
                tarf = f"{fname.split('/')[0]}_{fname.split('/')[1]}.tar"
                tarf = f"{ARCHIVE_PATH}/data/kepler/tpf/{fname.split('/')[0]}/{tarf}"
                tarfile.open(tarf, mode="r").extract(fname, tmpdir)
                tpfs.append(
                    lk.KeplerTargetPixelFile(
                        f"{tmpdir}/{fname}", quality_bitmask="none"
                    )
                )

        return lk.collections.TargetPixelFileCollection(tpfs)


# @profile
def do_components_plot(machine):
    tvec = (machine.P.poly_order + 1) * (len(machine.P.breaks) + 1)
    if machine.P.vectors.shape[1] == tvec:
        fig, ax = plt.subplots(1, 1, figsize=(7, 3))

        ax.plot(machine.P.vectors[:, :tvec])
        ax.set(title="Time Polynomials", xlabel="Time", ylabel="Strenght")
    else:
        if machine.P.focus:
            fvec = tvec + len(machine.P.breaks) + 1
            pvec = fvec
            fig, ax = plt.subplots(1, 3, figsize=(18, 3))

            ax[0].plot(machine.P.vectors[:, :tvec])
            ax[0].set(title="Time Polynomials", xlabel="Time", ylabel="Strenght")

            ax[1].plot(machine.P.vectors[:, tvec:fvec])
            ax[1].set(title="Focus", xlabel="Time")

            ax[2].plot(machine.P.vectors[:, pvec:])
            ax[2].set(
                title="PCA" if machine.P.other_vectors is None else "Positions",
                xlabel="Time",
            )
        else:
            pvec = tvec
            fig, ax = plt.subplots(1, 2, figsize=(12, 3))

            ax[0].plot(machine.P.vectors[:, :tvec])
            ax[0].set(title="Time Polynomials", xlabel="Time", ylabel="Strenght")

            ax[1].plot(machine.P.vectors[:, pvec:])
            ax[1].set(
                title="PCA" if machine.P.other_vectors is None else "Positions",
                xlabel="Time",
            )

    return fig


def plot_residuals_dash(mac):
    mean_f = np.log10(
        mac.uncontaminated_source_mask.astype(float)
        .multiply(mac.flux[mac.time_mask].mean(axis=0))
        .multiply(1 / mac.source_flux_estimates[:, None])
        .data
    )

    dx, dy = (
        mac.uncontaminated_source_mask.multiply(mac.dra),
        mac.uncontaminated_source_mask.multiply(mac.ddec),
    )
    dx = dx.data * 3600
    dy = dy.data * 3600
    phi, r = np.arctan2(dy, dx), np.hypot(dx, dy)

    A = _make_A_polar(
        phi,
        r,
        rmin=mac.rmin,
        rmax=mac.rmax,
        cut_r=mac.cut_r,
        n_r_knots=mac.n_r_knots,
        n_phi_knots=mac.n_phi_knots,
    )
    mean_f = 10 ** mean_f
    model = 10 ** A.dot(mac.psf_w)
    residuals = (model - mean_f) / mean_f

    source_flux = (
        mac.uncontaminated_source_mask.astype(float)
        .multiply(mac.source_flux_estimates[:, None])
        .data
    )
    source_col = mac.uncontaminated_source_mask.astype(float).multiply(mac.column).data
    source_row = mac.uncontaminated_source_mask.astype(float).multiply(mac.row).data

    fig, ax = plt.subplots(2, 2, figsize=(12, 9), facecolor="white")
    mad = stats.median_abs_deviation(residuals[np.isfinite(residuals)])

    ax[0, 0].scatter(
        residuals,
        source_flux,
        s=2,
        alpha=0.2,
        label=f"MAD = {mad:.3}",
    )
    ax[0, 0].legend(loc="upper right")
    ax[0, 0].set_ylabel("Gaia Source Flux", fontsize=12)
    ax[0, 0].set_xlabel("(F$_M$ - F$_D$)/F$_D$", fontsize=12)
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_ylim(1e3, 1e7)
    ax[0, 0].set_xlim(-1.01, 1.01)

    im = ax[0, 1].scatter(
        source_col, source_row, c=residuals, s=3, vmin=-1, vmax=1, alpha=1, cmap="RdBu"
    )
    ax[0, 1].set_ylabel("Pixel Row", fontsize=12)
    ax[0, 1].set_xlabel("Pixel Column", fontsize=12)

    ax[1, 0].scatter(
        source_flux, source_col, c=residuals, s=3, vmin=-1, vmax=1, alpha=1, cmap="RdBu"
    )
    ax[1, 0].set_xlabel("Gaia Source Flux", fontsize=12)
    ax[1, 0].set_ylabel("Pixel Column", fontsize=12)
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_xlim(1e3, 1e7)

    ax[1, 1].scatter(
        source_flux, source_row, c=residuals, s=3, vmin=-1, vmax=1, alpha=1, cmap="RdBu"
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label(label="(F$_M$ - F$_D$)/F$_D$", size="x-large")
    cbar.ax.tick_params(labelsize="large")
    ax[1, 1].set_xlabel("Gaia Source Flux", fontsize=12)
    ax[1, 1].set_ylabel("Pixel Row", fontsize=12)
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_xlim(1e3, 1e7)

    return fig


# @profile
def do_lcs(
    quarter=5,
    channel=1,
    batch_number=1,
    plot=True,
    dry_run=False,
    tar_lcs=False,
    tar_tpfs=True,
    fit_va=True,
    quiet=False,
    compute_node=False,
    augment_bkg=True,
    save_arrays="feather",
    iter_neg=True,
    use_cbv=True,
):

    ##############################################################################
    ############################# load TPF & config ##############################
    ##############################################################################

    with open(
        "%s/kepler_workflow/config/tpfmachine_keplerTPFs_config.yaml" % (PACKAGEDIR),
        "r",
    ) as f:
        config = yaml.safe_load(f)
    if quarter in [2, 12]:
        config["init"]["renormalize_tpf_bkg"] = False
    # get TPF file name list
    fname_list = get_file_list(quarter, channel, batch_number, tar_tpfs=tar_tpfs)
    if len(fname_list) < 50:
        logg.info(f"Warning: Actual batch size ({len(fname_list)}) is less than 50.")
    if dry_run:
        logg.info("Dry run!")
        sys.exit()
    # load TPFs
    logg.info("Loading TPFs from disk")
    if socket.gethostname().startswith("r"):
        sleep(np.random.randint(1, 30))
    tpfs = get_tpfs(fname_list, tar_tpfs=tar_tpfs)
    logg.info(f"Working with {len(tpfs)} TPFs")

    ##############################################################################
    ############################# do machine stuff ###############################
    ##############################################################################

    logg.info("Initializing PSFMachine")
    machine = pm.TPFMachine.from_TPFs(tpfs, **config["init"])
    if not compute_node:
        machine.quiet = quiet
    else:
        machine.quiet = True
        quiet = True
    logg.info("PSFMachine config:")
    logg.info(print_dict(config["init"]))

    del tpfs
    logg.info(machine)
    logg.info("PSFMachine time model config")
    logg.info(print_dict(config["time_model"]))

    # add mission background pixels
    date = machine.tpfs[0].path.split("/")[-1].split("-")[1].split("_")[0]
    bkg_file = (
        f"{ARCHIVE_PATH}/data/kepler/bkg"
        f"/{date[:4]}"
        f"/kplr{machine.tpfs[0].module:02}{machine.tpfs[0].output}-{date}_bkg.fits.gz"
    )
    print(bkg_file)
    if os.path.isfile(bkg_file) and augment_bkg:
        logg.info("Adding Mission BKG pixels...")
        logg.info(bkg_file)
        # read files
        mission_bkg_pixels = fitsio.read(bkg_file, columns=["RAWY", "RAWX"], ext=2)
        mission_bkg_data = fitsio.read(bkg_file, columns=["CADENCENO", "FLUX"], ext=1)

        # match cadences
        cadence_mask = np.in1d(machine.tpfs[0].time.jd, machine.time)
        cadenceno_machine = machine.tpfs[0].cadenceno[cadence_mask]
        mission_mask = np.in1d(mission_bkg_data["CADENCENO"], cadenceno_machine)

        keep_pix = (mission_bkg_pixels["RAWY"] > machine.row.min() - 50) & (
            mission_bkg_pixels["RAWY"] < machine.row.max() + 50
        )
        # get data
        data_augment = {
            "row": mission_bkg_pixels["RAWY"][keep_pix],
            "column": mission_bkg_pixels["RAWX"][keep_pix],
            "flux": mission_bkg_data["FLUX"][mission_mask][:, keep_pix],
        }
        del mission_bkg_pixels, mission_bkg_data
    else:
        data_augment = None

    logg.info("Building models...")
    # fit background
    machine.remove_background_model(
        plot=False,
        data_augment=data_augment,
    )

    # load shape model from FFI
    shape_model_path = (
        "%s/data/shape_models/ffi/ch%02i/%s_ffi_shape_model_ch%02i_q%02i.fits"
        % (
            PACKAGEDIR,
            channel,
            machine.tpf_meta["mission"][0],
            channel,
            quarter,
        )
    )
    if os.path.isfile(shape_model_path):
        try:
            machine.load_shape_model(input=shape_model_path, plot=False)
        except:
            logg.info("Loagind shape model failed, fitting PRF from data...")
            machine.build_shape_model(plot=False)
    else:
        logg.info("No shape model for this Q/Ch, fitting PRF from data...")
        machine.build_shape_model(**config["build_shape_model"])

    # SAP
    machine.compute_aperture_photometry(**config["compute_aperture_photometry"])
    # PSF phot

    # CBVs
    if use_cbv:
        ncomp = 4
        logg.info(f"Ussing CBVs first {ncomp} components")
        cbv_file = glob(
            f"{ARCHIVE_PATH}/data/kepler/cbv/"
            f"kplr*-q{machine.tpf_meta['quarter'][0]:02}-d25_lcbv.fits"
        )[0]
        if os.path.isfile(cbv_file):
            hdul = fits.open(cbv_file)
            ext = f"MODOUT_{machine.tpfs[0].module}_{machine.tpfs[0].output}"
            cbv_cdn = hdul[ext].data["CADENCENO"]
            cbv_vec = np.vstack(
                [hdul[ext].data[f"VECTOR_{i}"] for i in range(1, ncomp + 1)]
            )
        else:
            aux = download_kepler_cbvs(
                mission='Kepler',
                quarter=quarter,
                module=tpfs[0].module,
                output=tpfs[0].output,
            )
            cbv_cdn = aux["CADENCENO"]
            cbv_vec = np.vstack([aux[f"VECTOR_{i}"] for i in range(1, ncomp + 1)])

        # align cadences
        mask = np.isin(cbv_cdn, machine.cadenceno)
        cbv_cdn = cbv_cdn[mask]
        cbv_vec = cbv_vec[:, mask]
        cbv_vec_smooth = bspline_smooth(
            cbv_vec,
            x=machine.time,
            do_segments=True,
            n_knots=40,
        )
        other_vectors = (cbv_vec_smooth - cbv_vec_smooth.mean()) / (
            cbv_vec_smooth.max() - cbv_vec_smooth.mean()
        )
    else:
        other_vectors = None

    machine.build_time_model(**config["time_model"], other_vectors=other_vectors)
    logg.info("Fitting models...")
    machine.fit_model(fit_va=fit_va)
    if iter_neg:

        def find_neg_nns(negatives):
            # finds neighbors of negative sources
            neg_sources_ra = machine.sources.ra.values[negatives]
            neg_sources_dec = machine.sources.dec.values[negatives]

            # find contaminants
            all_sources_cat = SkyCoord(
                ra=machine.sources.ra * u.degree, dec=machine.sources.dec * u.degree
            )
            neg_sources_cat = SkyCoord(
                ra=neg_sources_ra * u.degree, dec=neg_sources_dec * u.degree
            )
            idx, d2d, _ = match_coordinates_3d(
                neg_sources_cat, all_sources_cat, nthneighbor=2
            )
            # nns within 5 arcsec
            neg_nns = idx[d2d.arcsec < 6]
            logg.info(f"Neg LCs: {negatives.shape[0]}")
            logg.info(f"Neg NNs: {neg_nns.shape[0]}")

            # return negtive and contaminatns idxs
            return np.unique(np.hstack([negatives, neg_nns])).ravel()

        # get sources with neg lightcurves
        negative_sources = (machine.ws_va < 0).sum(axis=0)  # > (0.02 * machine.nt)

        narrow_prior = find_neg_nns(np.where(negative_sources)[0])
        prior_sigma = (
            np.ones(machine.mean_model.shape[0])
            * 5
            * np.abs(machine.source_flux_estimates) ** 0.5
        )
        # we narrow the prior for negatives and their NNs
        prior_sigma[narrow_prior] /= 10
        machine.fit_model(fit_va=fit_va, prior_sigma=prior_sigma)

        # find remaining negatives and set them to zero.
        negative_sources = np.where((machine.ws_va < 0).sum(axis=0))[0]
        machine.ws_va[:, negative_sources] *= np.nan
        negative_sources = np.where((machine.ws < 0).sum(axis=0))[0]
        machine.ws[:, negative_sources] *= np.nan

    # compute source centroids
    machine.get_source_centroids(**config["get_source_centroids"])

    # get an index array to match the TPF cadenceno
    cadno_mask = np.in1d(machine.tpfs[0].time.jd, machine.time)
    # get KICs
    if compute_node or "kic" in machine.sources.columns:
        kics = machine.sources
    else:
        kics = get_KICs(machine.sources)

    # get the TPF index for each lc, a sources could fall in more than 1 tpf
    obs_per_pixel = machine.source_mask.multiply(machine.pix2obs).tocsr()
    tpf_idx = []
    for k in range(machine.source_mask.shape[0]):
        pix = obs_per_pixel[k].data
        mode = stats.mode(pix)[0]
        if len(mode) > 0:
            tpf_idx.append(mode[0])
        else:
            tpf_idx.append(
                [x for x, ss in enumerate(machine.tpf_meta["sources"]) if k in ss][0]
            )

    # get bkg light curves
    if config["init"]["renormalize_tpf_bkg"]:
        bkg_sap_flux = np.zeros((machine.flux.shape[0], machine.nsources))
        bkg_model = (
            machine.bkg_estimator.model[:, machine.pixels_in_tpf]
            + machine.bkg_median_level
        )
        for sdx in range(len(machine.aperture_mask)):
            bkg_sap_flux[:, sdx] = bkg_model[:, machine.aperture_mask[sdx]].sum(axis=1)
    else:
        bkg_sap_flux = np.zeros((machine.flux.shape[0], machine.nsources))

    # PSF residual light curves
    chi2_lc = np.zeros((machine.flux.shape[0], machine.nsources))
    residuals = ((machine.model_flux - machine.flux) / machine.flux_err) ** 2
    for sdx in range(len(machine.aperture_mask)):
        chi2_lc[:, sdx] = residuals[:, machine.source_mask[sdx].toarray()[0]].sum(
            axis=1
        )
    chi2_lc /= np.asarray(machine.source_mask.sum(axis=1)).flatten()

    # perturbed model metrics
    machine.get_psf_metrics()
    machine.source_psf_fraction /= np.nanpercentile(machine.source_psf_fraction, 99)
    machine.source_psf_fraction[machine.source_psf_fraction > 0.98] = 1

    # re-scale PSF to SAP level
    ratio = np.nanmean(machine.ws, axis=0) / np.nanmean(machine.sap_flux, axis=0)
    factor = np.nanmedian(ratio[machine.source_psf_fraction > 0.8])
    machine.ws /= factor
    machine.ws_va /= factor

    # get aperture mask in 2D shape
    aperture_mask_2d = aperture_mask_to_2d(
        machine.tpfs,
        machine.tpf_meta["sources"],
        machine.aperture_mask,
        machine.column,
        machine.row,
    )

    ##############################################################################
    ################################## save plots ################################
    ##############################################################################

    global_name = (
        "kbonus-%s-bkg_ch%02i_q%02i_v%s_lcs_bn%02i_fva%s_bkg%s_aug%s_sgm%s_ite%s_cbv%s"
        % (
            machine.tpf_meta["mission"][0].lower(),
            channel,
            quarter,
            lc_version,
            batch_number,
            str(fit_va)[0],
            str(config["init"]["renormalize_tpf_bkg"])[0],
            str(augment_bkg)[0],
            str(config["time_model"]["segments"])[0],
            str(iter_neg)[0],
            str(use_cbv)[0],
        )
    )
    if not plot:
        plot = np.random.choice([False] * 25 + [True])

    if plot:
        dir_name = "%s/figures/tpf/ch%02i" % (OUTPUT_PATH, channel)
        logg.info(f"Saving diagnostic plots into: {dir_name}")
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        file_name = f"{dir_name}/{global_name}.pdf"

        with PdfPages(file_name) as pages:
            # BKG figures
            if config["init"]["renormalize_tpf_bkg"]:
                FigureCanvasPdf(machine.bkg_estimator.plot()).print_figure(pages)
                FigureCanvasPdf(
                    machine.plot_background_model(frame_index=machine.nt // 2)
                ).print_figure(pages)
            # SHAPE FIGURE
            FigureCanvasPdf(machine.plot_shape_model()).print_figure(pages)
            FigureCanvasPdf(plot_residuals_dash(machine)).print_figure(pages)
            # TIME FIGURE
            if fit_va:
                try:
                    time_fig1, time_fig2 = machine.plot_time_model()
                    if machine.cartesian_knot_spacing == "sqrt":
                        xknots = np.linspace(
                            -np.sqrt(machine.time_radius),
                            np.sqrt(machine.time_radius),
                            machine.time_nknots,
                        )
                        xknots = np.sign(xknots) * xknots ** 2
                    else:
                        xknots = np.linspace(
                            -machine.time_radius,
                            machine.time_radius,
                            machine.time_nknots,
                        )
                    xknots, yknots = np.meshgrid(xknots, xknots)
                    time_fig1.axes[-2].scatter(xknots, yknots, c="k", s=2, marker="x")
                    FigureCanvasPdf(time_fig1).print_figure(pages)
                    FigureCanvasPdf(time_fig2).print_figure(pages)
                except:
                    pass
                FigureCanvasPdf(do_components_plot(machine)).print_figure(pages)

        plt.close()

    ##############################################################################
    ################################## save lcs ##################################
    ##############################################################################

    dir_name = "%s/%s/ch%02i/q%02i" % (
        LCS_PATH,
        machine.tpf_meta["mission"][0].lower(),
        channel,
        quarter,
    )
    logg.info(f"Saving light curves into: {dir_name}")
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    tarf_name = f"{dir_name}/{global_name}.tar.gz"
    if tar_lcs:
        logg.info("LCFs will be tar.gz")
        tar = tarfile.open(tarf_name, mode="w:gz")
    centroids_row, centroids_col = [], []
    for idx, srow in tqdm(
        machine.sources.iterrows(),
        total=machine.nsources,
        desc="Saving LCFs",
        disable=quiet,
    ):
        # we skip empty lcs
        if (
            np.isnan(machine.ws_va[:, idx]).all()
            and np.isnan(machine.sap_flux[:, idx]).all()
        ) or (
            np.isnan(machine.ws_va[:, idx]).all()
            and (machine.sap_flux[:, idx] == 0).all()
        ):
            continue

        # gather data
        data = {
            "cadenceno": machine.tpfs[tpf_idx[idx]].cadenceno[cadno_mask],
            "time": machine.time,
            "flux": machine.ws_va[:, idx] if fit_va else machine.ws[:, idx],
            "flux_err": machine.werrs_va[:, idx] if fit_va else machine.werrs[:, idx],
            "sap_flux": machine.sap_flux[:, idx],
            "sap_flux_err": machine.sap_flux_err[:, idx],
            "centroid_col": vars(machine)[
                "source_centroids_column_%s"
                % (config["get_source_centroids"]["method"])
            ][idx],
            "centroid_row": vars(machine)[
                "source_centroids_row_%s" % (config["get_source_centroids"]["method"])
            ][idx],
            "quality": machine.tpfs[tpf_idx[idx]].quality[cadno_mask],
            "psf_flux_NVA": machine.ws[:, idx],
            "psf_flux_err_NVA": machine.werrs[:, idx],
            "sap_bkg": bkg_sap_flux[:, idx],
            "red_chi2": chi2_lc[:, idx],
        }
        centroids_row.append(np.nanmean(data["centroid_row"]).value)
        centroids_col.append(np.nanmean(data["centroid_col"]).value)
        # get LC metadata
        lc_meta = machine._make_meta_dict(idx, srow, True)

        # get extra metadata
        extra_meta = {
            "PIXINAP": machine.aperture_mask[idx].sum(),
            "KEPLERID": kics["kic"][idx],
            "KEPMAG": kics["kepmag"][idx],
            "TPFORG": machine.tpfs[tpf_idx[idx]].meta["KEPLERID"],
            "TELESCOP": machine.tpfs[0].meta["TELESCOP"],
            "INSTRUME": machine.tpfs[0].meta["INSTRUME"],
            "OBSMODE": machine.tpfs[0].meta["OBSMODE"],
            "OUTPUT": machine.tpfs[0].meta["OUTPUT"],
            "SEASON": machine.tpfs[0].meta["SEASON"],
            "EQUINOX": machine.tpfs[0].meta["EQUINOX"],
            "GAIA_DES": srow.designation,
            "PSFFRAC": machine.source_psf_fraction[idx],
            "PERRATIO": machine.perturbed_ratio_mean[idx],
            "PERSTD": machine.perturbed_std[idx],
            "ITERNEG": 1
            if np.isnan(machine.ws[:, idx]).all()
            and not np.isnan(machine.ws_va[:, idx]).all()
            else 0,
        }

        try:
            aperture_mask = aperture_mask_2d[f"{tpf_idx[idx]}_{idx}"]
        except KeyError:
            aperture_mask = None

        hdul = make_hdul(data, lc_meta, extra_meta, aperture_mask=aperture_mask)
        if "KIC" in hdul[0].header["LABEL"]:
            target_name = f"KIC-{int(hdul[0].header['LABEL'].split(' ')[-1]):09}"
        else:
            target_name = hdul[0].header["LABEL"].replace(" ", "-")
        fname = "hlsp_kbonus-bkg_kepler_kepler_%s-q%02i_kepler_v%s_lc.fits" % (
            target_name.lower(),
            quarter,
            lc_version,
        )
        if tar_lcs:
            with tempfile.NamedTemporaryFile(mode="wb") as tmp:
                hdul.writeto(tmp)
                tar.add(tmp.name, arcname=fname)
        else:
            hdul.writeto("%s/%s.gz" % (dir_name, fname), overwrite=True, checksum=True)
        del hdul

    if tar_lcs:
        tar.close()

    if save_arrays == "npz":
        np.savez(
            f"{dir_name}/{global_name}.npz",
            time=machine.time,
            cadence=machine.cadenceno,
            flux=machine.ws_va,
            flux_err=machine.werrs_va,
            sap_flux=machine.sap_flux,
            sap_flux_err=machine.sap_flux_err,
            psfnva_flux=machine.ws,
            psfnva_flux_err=machine.werrs,
            chi2=chi2_lc,
            sources=machine.sources.designation.values,
            column=np.nanmean(data["centroid_col"]),
            row=np.nanmean(data["centroid_row"]),
            ra=machine.sources.ra,
            dec=machine.sources.dec,
        )
    elif save_arrays == "feather":
        index = pd.MultiIndex.from_arrays(
            [machine.cadenceno, machine.time], names=["cedence", "jd"]
        )
        locs = pd.DataFrame(
            [
                machine.sources.ra.values,
                machine.sources.dec.values,
                np.array(centroids_col),
                np.array(centroids_row),
                machine.sources.tpf_id.values,
            ],
            index=["ra", "dec", "column", "row", "tpf_id"],
            columns=machine.sources.designation.values,
        ).T
        fname = f"{dir_name}/{global_name}.coord.feather"
        locs.reset_index().to_feather(fname)

        for name, val, val_err in zip(
            ["psf", "novapsf", "sap", "chi2"],
            [machine.ws_va, machine.ws, machine.sap_flux, chi2_lc],
            [machine.werrs_va, machine.werrs, machine.sap_flux_err, None],
        ):
            fname = f"{dir_name}/{global_name}.{name}.feather"
            df = pd.DataFrame(
                val, index=index, columns=machine.sources.designation.values
            )
            df.reset_index().to_feather(fname)
            if not val_err is None:
                fname = f"{dir_name}/{global_name}.{name}_err.feather"
                df = pd.DataFrame(
                    val_err, index=index, columns=machine.sources.designation.values
                )
                df.reset_index().to_feather(fname)
    else:
        raise ValueError("Wrong type of array files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoEncoder")
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
    parser.add_argument(
        "--batch-index",
        dest="batch_index",
        type=int,
        default=-1,
        help="File with index list of channel/quarter/batch size/batch number",
    )
    parser.add_argument(
        "--batch-number",
        dest="batch_number",
        type=int,
        default=1,
        help="Batch number or batch index in --batch-file",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=False,
        help="Make diagnostic plots.",
    )
    parser.add_argument(
        "--fit-va",
        dest="fit_va",
        action="store_true",
        default=False,
        help="Fit Velocity aberration.",
    )
    parser.add_argument(
        "--augment-bkg",
        dest="augment_bkg",
        action="store_true",
        default=False,
        help="Augment background pixels.",
    )
    parser.add_argument(
        "--tar-lcs",
        dest="tar_lcs",
        action="store_true",
        default=False,
        help="Create a tar.gz file with light curves.",
    )
    parser.add_argument(
        "--tar-tpfs",
        dest="tar_tpfs",
        action="store_true",
        default=False,
        help="Is archive in tarball files.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Dry run.",
    )
    parser.add_argument(
        "--force-log",
        dest="force_log",
        action="store_true",
        default=False,
        help="Forcel logging.",
    )
    parser.add_argument(
        "--save-arrays",
        dest="save_arrays",
        type=str,
        default="feather",
        help="Save W's as npy files for quick access.",
    )
    parser.add_argument(
        "--iter-neg",
        dest="iter_neg",
        action="store_true",
        default=False,
        help="Iter negative light curves.",
    )
    parser.add_argument(
        "--use-cbv",
        dest="use_cbv",
        action="store_true",
        default=False,
        help="Iter negative light curves.",
    )
    parser.add_argument("--log", dest="log", default=0, help="Logging level")
    args = parser.parse_args()
    # set verbose level for logger
    try:
        args.log = int(args.log)
    except:
        args.log = str(args.log.upper())

    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # send to log file when running in compute nodes
    if (
        (socket.gethostname() in ["NASAs-MacBook-Pro.local"])
        or (socket.gethostname()[:3] == "pfe")
        or (args.force_log)
    ):
        if socket.gethostname() in ["NASAs-MacBook-Pro.local"]:
            compute_node = False
        else:
            compute_node = True
        hand = logging.StreamHandler(sys.stdout)
        hand.setFormatter(logging.Formatter(FORMAT))
    else:
        compute_node = True
        hand = logging.FileHandler(
            f"{PACKAGEDIR}/logs/make_lightcurve_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.info"
        )
        hand.setLevel(logging.INFO)
    hand.setFormatter(logging.Formatter(FORMAT))
    logg.handlers.clear()
    logg.addHandler(hand)
    logg.setLevel(args.log)
    logg.propagate = False

    if args.channel is None and args.batch_index > -1:
        batch_info = "%s/data/support/kepler_batch_info_quarter%i.dat" % (
            PACKAGEDIR,
            args.quarter,
        )
        logg.info(f"Batch info file {batch_info}")
        params = np.loadtxt(batch_info, dtype=int, delimiter=" ", comments="#")
        args.quarter = params[args.batch_index - 1, 1]
        args.channel = params[args.batch_index - 1, 2]
        args.batch_total = params[args.batch_index - 1, 3]
        args.batch_number = params[args.batch_index - 1, 4]

    logg.info("Program config")
    logg.info(print_dict(vars(args)))
    if args.dry_run:
        logg.info("Dry run!")
        sys.exit()

    kwargs = vars(args)
    del kwargs["force_log"]
    try:
        del kwargs["batch_index"], kwargs["batch_total"]
    except KeyError:
        pass
    kwargs["quiet"] = True if kwargs.pop("log") in [0, "0", "NOTSET"] else False

    do_lcs(**kwargs, compute_node=compute_node)
    logg.info("Done!")
