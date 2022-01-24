import os
import sys
import argparse
import socket
import yaml
import tarfile
import tempfile
import logging
from datetime import datetime
from time import sleep
import numpy as np
import pandas as pd
import psfmachine as pm
import lightkurve as lk
from scipy import stats
from tqdm import tqdm
from memory_profiler import profile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
from astropy.io import fits
import fitsio

# from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_3d
import astropy.units as u
from astroquery.vizier import Vizier

from paths import ARCHIVE_PATH, OUTPUT_PATH, LCS_PATH, PACKAGEDIR

import warnings
from scipy import sparse

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)

logg = logging.getLogger(__name__)
lc_version = "1.0"

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
def get_file_list(quarter, channel, batch_size, batch_number, tar_tpfs=True):

    lookup_table = pd.read_csv(
        "%s/data/support/kepler_tpf_map_q%02i%s.csv"
        % (PACKAGEDIR, quarter, "_tar" if tar_tpfs else ""),
        index_col=0,
    )
    files_in = lookup_table.query(
        "channel == %i and quarter == %i" % (channel, quarter)
    )
    if files_in.shape[0] == 0:
        raise IndexError("Channel does not contain TPFs.")
    if batch_size > 0:
        files_in = files_in.iloc[
            batch_size * (batch_number - 1) : batch_size * (batch_number)
        ]
        if files_in.shape[0] == 0:
            raise IndexError("Batch does not contain TPFs.")
        return files_in.file_name.tolist()
    else:
        return files_in.file_name.tolist()


# @profile
def make_hdul(data, lc_meta, extra_meta, fit_va=True):
    meta = {
        "ORIGIN": lc_meta["ORIGIN"],
        "VERSION": pm.__version__,
        "APERTURE": lc_meta["APERTURE"],
        # telescope info
        "MISSION": lc_meta["MISSION"],
        "TELESCOP": extra_meta["TELESCOP"],
        "INSTRUME": extra_meta["INSTRUME"],
        "OBSMODE": extra_meta["OBSMODE"],
        "SEASON": extra_meta["SEASON"],
        "CHANNEL": lc_meta["CHANNEL"],
        "MODULE": lc_meta["MODULE"],
        "OUTPUT": extra_meta["OUTPUT"],
        "QUARTER": lc_meta["QUARTER"],
        # "CAMPAIGN": lc_meta["CAMPAIGN"],
        # objct info
        "LABEL": f"KIC {extra_meta['KEPLERID']}"
        if extra_meta["KEPLERID"] != 0
        else extra_meta["GAIA_DES"],
        "TARGETID": lc_meta["TARGETID"],
        "RA_OBJ": lc_meta["RA"],
        "DEC_OBJ": lc_meta["DEC"],
        "EQUINOX": extra_meta["EQUINOX"],
        # KIC info
        "KEPLERID": extra_meta["KEPLERID"] if extra_meta["KEPLERID"] != 0 else "",
        "KEPMAG": extra_meta["KEPMAG"] if np.isfinite(extra_meta["KEPMAG"]) else "",
        "TPFORG": extra_meta["TPFORG"],
        # gaia catalog info
        "GAIAID": extra_meta["GAIA_DES"],
        "PMRA": lc_meta["PMRA"] if np.isfinite(lc_meta["PMRA"]) else "",
        "PMDEC": lc_meta["PMDEC"] if np.isfinite(lc_meta["PMDEC"]) else "",
        "PARALLAX": lc_meta["PARALLAX"] if np.isfinite(lc_meta["PARALLAX"]) else "",
        "GMAG": lc_meta["GMAG"] if np.isfinite(lc_meta["GMAG"]) else "",
        "RPMAG": lc_meta["RPMAG"] if np.isfinite(lc_meta["RPMAG"]) else "",
        "BPMAG": lc_meta["BPMAG"] if np.isfinite(lc_meta["BPMAG"]) else "",
        # extraction info
        "TPF_ROW": lc_meta["ROW"],
        "TPF_COL": lc_meta["COLUMN"],
        "SAP": lc_meta["SAP"],
        "ROW": np.nanmean(data["centroid_row"]).value,
        "COLUMN": np.nanmean(data["centroid_col"]).value,
        "FLFRCSAP": lc_meta["FLFRCSAP"] if np.isfinite(lc_meta["FLFRCSAP"]) else "",
        "CROWDSAP": lc_meta["CROWDSAP"] if np.isfinite(lc_meta["CROWDSAP"]) else "",
        "NPIXSAP": extra_meta["PIXINAP"],
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
    }
    if fit_va:
        lc_dct["psf_flux_nvs"] = data["psf_flux_NVA"]
        lc_dct["psf_flux_err_nvs"] = data["psf_flux_err_NVA"]

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
        hdul[0].header[key] = val

    return hdul


# @profile
def get_tpfs(fname_list, tar_tpfs=True):
    if not tar_tpfs:
        return lk.collections.TargetPixelFileCollection(
            [lk.read(f) for f in fname_list]
        )
    else:
        tpfs = []
        with tempfile.TemporaryDirectory(prefix="temp_fits") as tmpdir:
            for k, fname in enumerate(fname_list):
                tarf = f"{fname.split('/')[0]}_{fname.split('/')[1]}.tar"
                tarf = f"{ARCHIVE_PATH}/data/kepler/tpf/{fname.split('/')[0]}/{tarf}"
                tarfile.open(tarf, mode="r").extract(fname, tmpdir)
                tpfs.append(lk.read(f"{tmpdir}/{fname}"))

        return lk.collections.TargetPixelFileCollection(tpfs)


# @profile
def do_poscorr_plot(machine):
    (
        time_original,
        time_binned,
        flux_binned_raw,
        flux_binned,
        flux_err_binned,
        poscorr1_smooth,
        poscorr2_smooth,
        poscorr1_binned,
        poscorr2_binned,
    ) = machine._time_bin(npoints=machine.n_time_points)
    fig, ax = plt.subplots(1, 2, figsize=(15, 3))

    ax[0].set_title("Poscorr 1")
    ax[0].plot(
        time_original,
        np.median(machine.pos_corr1, axis=0),
        c="k",
        lw=1.2,
        label="Median",
    )
    ax[0].plot(time_original, poscorr1_smooth, c="r", ls="-", lw=0.8, label="Smooth")
    ax[0].plot(
        time_binned[:, 0],
        poscorr1_binned[:, 0],
        c="g",
        marker="o",
        lw=0,
        ms=5,
        label="Knots",
    )
    ax[0].set_xlabel("Time (whitened)")
    ax[0].set_ylabel("Mean")
    ax[0].legend(loc="best")

    ax[1].set_title("Poscorr 2")
    ax[1].plot(time_original, np.median(machine.pos_corr2, axis=0), c="k", lw=1.2)
    ax[1].plot(time_original, poscorr2_smooth, c="r", ls="-", lw=0.8)
    ax[1].plot(time_binned[:, 0], poscorr2_binned[:, 0], c="g", marker="o", lw=0, ms=5)
    ax[1].set_xlabel("Time (whitened)")
    fig.tight_layout()

    return fig


# @profile
def do_lcs(
    quarter=5,
    channel=1,
    batch_size=50,
    batch_number=1,
    plot=True,
    dry_run=False,
    tar_lcs=False,
    tar_tpfs=True,
    fit_va=True,
    quiet=False,
    compute_node=False,
    augment_bkg=True,
    save_npy=False,
):

    ##############################################################################
    ############################# load TPF & config ##############################
    ##############################################################################

    with open(
        "%s/kepler_workflow/config/tpfmachine_keplerTPFs_config.yaml" % (PACKAGEDIR),
        "r",
    ) as f:
        config = yaml.safe_load(f)
    # get TPF file name list
    fname_list = get_file_list(
        quarter, channel, batch_size, batch_number, tar_tpfs=tar_tpfs
    )
    if len(fname_list) < batch_size:
        logg.info(
            f"Warning: Actual batch size ({len(fname_list)}) "
            f"is smaller than asked ({batch_size})."
        )
    if len(fname_list) < 50:
        logg.info(f"Warning: Actual batch size ({len(fname_list)}) is less than 50.")
    if dry_run:
        logg.info("Dry run!")
        sys.exit()
    # load TPFs
    logg.info("Loading TPFs from disk")
    if socket.gethostname().startswith("r"):
        sleep(np.random.randint(10, 30))
    tpfs = get_tpfs(fname_list, tar_tpfs=tar_tpfs)

    ##############################################################################
    ############################# do machine stuff ###############################
    ##############################################################################

    logg.info("Initializing PSFMachine")
    machine = pm.TPFMachine.from_TPFs(tpfs, **config)
    if not compute_node:
        machine.quiet = quiet
    else:
        machine.quiet = True
        quiet = True
    logg.info("PSFMachine config:")
    print_dict(config)

    del tpfs
    logg.info(machine)

    # add mission background pixels
    date = machine.tpfs[0].path.split("/")[-1].split("-")[1].split("_")[0]
    bkg_file = (
        f"{ARCHIVE_PATH}/data/kepler/bkg"
        f"/{date[:4]}"
        f"/kplr{machine.tpfs[0].module:02}{machine.tpfs[0].output}-{date}_bkg.fits.gz"
    )
    if os.path.isfile(bkg_file) and augment_bkg:
        logg.info("Adding Mission BKG pixels...")
        logg.info(bkg_file)
        # read files
        # mission_bkg_pixels = Table.read(bkg_file, hdu=2)
        mission_bkg_pixels = fitsio.read(bkg_file, columns=["RAWY", "RAWX"], ext=2)
        # mission_bkg_data = Table.read(bkg_file, hdu=1)
        mission_bkg_data = fitsio.read(bkg_file, columns=["CADENCENO", "FLUX"], ext=1)

        # match cadences
        cadence_mask = np.in1d(machine.tpfs[0].time.jd, machine.time)
        cadenceno_machine = machine.tpfs[0].cadenceno[cadence_mask]
        mission_mask = np.in1d(mission_bkg_data["CADENCENO"], cadenceno_machine)
        # get data
        data_augment = {
            "row": mission_bkg_pixels["RAWY"],
            "column": mission_bkg_pixels["RAWX"],
            "flux": mission_bkg_data["FLUX"][mission_mask],
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
        machine.load_shape_model(input=shape_model_path, plot=False)
    else:
        logg.info("No shape model for this Q/Ch, fitting PRF from data...")
        machine.build_shape_model(plot=plot)

    # SAP
    machine.compute_aperture_photometry(
        aperture_size="optimal", target_complete=1, target_crowd=1
    )
    # PSF phot
    machine.build_time_model(plot=False)
    logg.info("Fitting models...")
    machine.fit_model(fit_va=fit_va)
    iter_negative = False
    if iter_negative:
        # More than 2% negative cadences
        negative_sources = (machine.ws_va < 0).sum(axis=0) > (0.02 * machine.nt)
        idx = 1
        while len(negative_sources) > 0:
            machine.mean_model[negative_sources] *= 0
            machine.fit_model(fit_va=fit_va)
            negative_sources = np.where((machine.ws_va < 0).all(axis=0))[0]
            idx += 1
            if idx >= 3:
                break

    # compute source centroids
    centroid_method = "scene"
    machine.get_source_centroids(method=centroid_method)

    # get an index array to match the TPF cadenceno
    cadno_mask = np.in1d(machine.tpfs[0].time.jd, machine.time)
    # get KICs
    if not compute_node:
        kics = get_KICs(machine.sources)
    else:
        kics = machine.sources

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

    ##############################################################################
    ################################## save plots ################################
    ##############################################################################

    if plot:
        dir_name = "%s/figures/tpf/ch%02i" % (OUTPUT_PATH, channel)
        logg.info(f"Saving diagnostic plots into: {dir_name}")
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        file_name = "%s/%s_models_ch%02i_q%02i_b%03i-%02i_%s_%s_tk%i_tp%i_bkg%s.pdf" % (
            dir_name,
            machine.tpf_meta["mission"][0],
            channel,
            quarter,
            batch_size,
            batch_number,
            machine.time_corrector.replace("_", ""),
            machine.cartesian_knot_spacing,
            machine.n_time_knots,
            machine.n_time_points,
            str(config["renormalize_tpf_bkg"])[0],
        )

        # bkg model
        if config["renormalize_tpf_bkg"]:
            bkg_fig = machine.plot_background_model(frame_index=machine.nt // 2)
            bkg_fig_2 = machine.bkg_est.plot()

        # SHAPE FIGURE
        shape_fig = machine.plot_shape_model()

        # TIME FIGURE
        if fit_va:
            try:
                time_fig = machine.plot_time_model()
                if machine.cartesian_knot_spacing == "sqrt":
                    xknots = np.linspace(
                        -np.sqrt(machine.time_radius),
                        np.sqrt(machine.time_radius),
                        machine.n_time_knots,
                    )
                    xknots = np.sign(xknots) * xknots ** 2
                else:
                    xknots = np.linspace(
                        -machine.time_radius, machine.time_radius, machine.n_time_knots
                    )
                xknots, yknots = np.meshgrid(xknots, xknots)
                time_fig.axes[-2].scatter(xknots, yknots, c="k", s=2, marker="x")
                time_fig.suptitle(f"Time model: {machine.time_corrector}")
            except:
                pass

        with PdfPages(file_name) as pages:
            if config["renormalize_tpf_bkg"]:
                FigureCanvasPdf(bkg_fig_2).print_figure(pages)
                FigureCanvasPdf(bkg_fig).print_figure(pages)
            FigureCanvasPdf(shape_fig).print_figure(pages)
            if fit_va:
                try:
                    FigureCanvasPdf(time_fig).print_figure(pages)
                except:
                    pass
            if fit_va and machine.time_corrector == "pos_corr":
                FigureCanvasPdf(do_poscorr_plot(machine)).print_figure(pages)
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
    if tar_lcs:
        logg.info("LCFs will be tar.gz")
        tar = tarfile.open(
            "%s/kbonus-bkgd_ch%02i_q%02i_v%s_lc_b%03i-%02i_%s_%s_tk%i_tp%i_fva%s_bkg%s_aug%s.tar.gz"
            % (
                dir_name,
                channel,
                quarter,
                lc_version,
                batch_size,
                batch_number,
                machine.time_corrector.replace("_", ""),
                machine.cartesian_knot_spacing,
                machine.n_time_knots,
                machine.n_time_points,
                str(fit_va)[0],
                str(config["renormalize_tpf_bkg"])[0],
                str(augment_bkg)[0],
            ),
            mode="w:gz",
        )
    for idx, srow in tqdm(
        machine.sources.iterrows(),
        total=machine.nsources,
        desc="Saving LCFs",
        disable=quiet,
    ):
        # we skipt empty lcs
        if (
            np.isnan(machine.ws[:, idx]).all()
            and np.isnan(machine.sap_flux[:, idx]).all()
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
                "source_centroids_column_%s" % (centroid_method)
            ][idx],
            "centroid_row": vars(machine)[
                "source_centroids_row_%s" % (centroid_method)
            ][idx],
            "quality": machine.tpfs[tpf_idx[idx]].quality[cadno_mask],
            "psf_flux_NVA": machine.ws[:, idx],
            "psf_flux_err_NVA": machine.werrs[:, idx],
        }
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
        }

        hdul = make_hdul(data, lc_meta, extra_meta, fit_va=fit_va)
        if "KIC" in hdul[0].header["LABEL"]:
            target_name = f"KIC-{int(hdul[0].header['LABEL'].split(' ')[-1]):09}"
        else:
            target_name = hdul[0].header["LABEL"].replace(" ", "-")
        fname = "hlsp_kbonus-kbkgd_kepler_kepler_%s-q%02i_kepler_v%s_lc.fits" % (
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

    if tar_lcs:
        tar.close()

    if save_npy:
        np.savez(
            tar.replace("tar.gz", "npz"),
            time=machine.time,
            flux=machine.flux,
            flux_err=machine.flux_err,
            source=machine.sources.designation.values,
        )


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
        "--batch-size",
        dest="batch_size",
        type=int,
        default=50,
        help="Batch size",
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
        "--save-npy",
        dest="save_npy",
        action="store_true",
        default=False,
        help="Save W's as npy files for quick access.",
    )
    parser.add_argument("--log", dest="log", default=0, help="Logging level")
    args = parser.parse_args()
    # set verbose level for logger
    try:
        args.log = int(args.log)
    except:
        args.log = str(args.log.upper())

    FORMAT = "%(filename)s:%(lineno)s : %(message)s"
    # send to log file when running in compute nodes
    if (
        (socket.gethostname() in ["NASAs-MacBook-Pro.local"])
        or (socket.gethostname()[:3] == "pfe")
        or (args.force_log)
    ):
        if args.force_log:
            compute_node = True
        else:
            compute_node = False
        hand = logging.StreamHandler(sys.stdout)
        hand.setFormatter(logging.Formatter(FORMAT))
    else:
        compute_node = True
        hand = logging.FileHandler(
            f"{PACKAGEDIR}/logs/make_lightcurve_{os.getpid()}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.info"
        )
        hand.setLevel(logging.INFO)
    hand.setFormatter(logging.Formatter(FORMAT))
    logg.addHandler(hand)
    logg.setLevel(args.log)

    if args.channel is None and args.batch_index > -1:
        batch_info = "%s/data/support/kepler_batch_info_quarter%i.dat" % (
            PACKAGEDIR,
            args.quarter,
        )
        logg.info(f"Batch info file {batch_info}")
        params = np.loadtxt(batch_info, dtype=int, delimiter=" ", comments="#")
        args.quarter = params[args.batch_index - 1, 1]
        args.channel = params[args.batch_index - 1, 2]
        args.batch_size = params[args.batch_index - 1, 3]
        args.batch_total = params[args.batch_index - 1, 4]
        args.batch_number = params[args.batch_index - 1, 5]

    print_dict(vars(args))
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
