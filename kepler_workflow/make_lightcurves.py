import os
import sys
import argparse
import yaml
import tarfile
import tempfile
import warnings
import logging
import numpy as np
import pandas as pd
import psfmachine as pm
import lightkurve as lk
from scipy import sparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages

from astropy.coordinates import SkyCoord, match_coordinates_3d
import astropy.units as u
from astroquery.vizier import Vizier

from paths import ARCHIVE_PATH, OUTPUT_PATH, LCS_PATH, PACKAGEDIR

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)

log = logging.getLogger(__name__)
lc_version = "1.0"

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
    catalog.loc[:, "kic"] = ""
    catalog.loc[mdist.arcsec < 2, "kic"] = kic[midx[mdist.arcsec < 2]]["KIC"].data.data
    catalog.loc[:, "kepmag"] = None
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
    files_in = files_in.iloc[
        batch_size * (batch_number - 1) : batch_size * (batch_number)
    ]
    if files_in.shape[0] == 0:
        raise IndexError("Batch does not contain TPFs.")
    return files_in.file_name.tolist()


# @profile
def make_hdul(lc, catalog, extra_meta, fit_va=True):
    meta = {
        "ORIGIN": lc.meta["ORIGIN"],
        "VERSION": pm.__version__,
        "APERTURE": lc.meta["APERTURE"],
        # telescope info
        "MISSION": lc.meta["MISSION"],
        "TELESCOP": extra_meta["TELESCOP"],
        "INSTRUME": extra_meta["INSTRUME"],
        "OBSMODE": extra_meta["OBSMODE"],
        "SEASON": extra_meta["OBSMODE"],
        "CHANNEL": lc.meta["CHANNEL"],
        "MODULE": lc.meta["MODULE"],
        "OUTPUT": extra_meta["OUTPUT"],
        "QUARTER": lc.meta["QUARTER"],
        # "CAMPAIGN": lc.meta["CAMPAIGN"],
        # objct info
        "LABEL": "KIC %s" % (extra_meta["KEPLERID"])
        if extra_meta["KEPLERID"] != ""
        else catalog.designation,
        "TARGETID": lc.meta["TARGETID"],
        "RA_OBJ": lc.meta["RA"],
        "DEC_OBJ": lc.meta["DEC"],
        "EQUINOX": extra_meta["EQUINOX"],
        # KIC info
        "KEPLERID": extra_meta["KEPLERID"],
        "TPFORG": extra_meta["TPFORG"],
        # gaia catalog info
        "GAIAID": catalog.designation,
        "PMRA": lc.meta["PMRA"] if np.isfinite(lc.meta["PMRA"]) else "",
        "PMDEC": lc.meta["PMDEC"] if np.isfinite(lc.meta["PMDEC"]) else "",
        "PARALLAX": lc.meta["PARALLAX"] if np.isfinite(lc.meta["PARALLAX"]) else "",
        "GMAG": lc.meta["GMAG"] if np.isfinite(lc.meta["GMAG"]) else "",
        "RPMAG": lc.meta["RPMAG"] if np.isfinite(lc.meta["RPMAG"]) else "",
        "BPMAG": lc.meta["BPMAG"] if np.isfinite(lc.meta["BPMAG"]) else "",
        # extraction info
        "SAP": lc.meta["SAP"],
        "ROW": lc.meta["ROW"],
        "COLUMN": lc.meta["COLUMN"],
        "FLFRCSAP": lc.meta["FLFRCSAP"] if np.isfinite(lc.meta["FLFRCSAP"]) else "",
        "CROWDSAP": lc.meta["CROWDSAP"] if np.isfinite(lc.meta["CROWDSAP"]) else "",
        "NPIXSAP": extra_meta["PIXINAP"],
    }
    if fit_va:
        lc_dct = {
            "cadenceno": extra_meta["cadenceno"],
            "time": lc.time - 2454833,
            "flux": lc.flux.value * (u.electron / u.second),
            "flux_err": lc.flux_err.value * (u.electron / u.second),
            "sap_flux": lc.sap_flux.value * (u.electron / u.second),
            "sap_flux_err": lc.sap_flux_err.value * (u.electron / u.second),
            "psf_flux_nvs": lc.psf_flux_NVA.value * (u.electron / u.second),
            "psf_flux_err_nvs": lc.psf_flux_err_NVA.value * (u.electron / u.second),
            "quality": extra_meta["quality"],
            "centroid_column": extra_meta["centroid_col"] * u.pixel,
            "centroid_row": extra_meta["centroid_row"] * u.pixel,
        }
    else:
        lc_dct = {
            "cadenceno": extra_meta["cadenceno"],
            "time": lc.time - 2454833,
            "flux": lc.flux.value * (u.electron / u.second),
            "flux_err": lc.flux_err.value * (u.electron / u.second),
            "sap_flux": lc.sap_flux.value * (u.electron / u.second),
            "sap_flux_err": lc.sap_flux_err.value * (u.electron / u.second),
            "quality": extra_meta["quality"],
            "centroid_column": extra_meta["centroid_col"] * u.pixel,
            "centroid_row": extra_meta["centroid_row"] * u.pixel,
        }
    lc_ = lk.LightCurve(lc_dct, meta=meta)
    del lc_dct["time"], lc_dct["flux_err"]
    hdul = lc_.to_fits(**lc_dct, **lc_.meta)
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
):

    # load config file for TPFs
    with open(
        "%s/kepler_workflow/tpfmachine_keplerTPFs_config.yaml" % (PACKAGEDIR), "r"
    ) as f:
        config = yaml.safe_load(f)
    # get TPF file name list
    fname_list = get_file_list(
        quarter, channel, batch_size, batch_number, tar_tpfs=tar_tpfs
    )
    if len(fname_list) < batch_size:
        log.info(
            f"Warning: Actual batch size ({len(fname_list)}) "
            f"is smaller than asked ({batch_size})."
        )
    if len(fname_list) < 50:
        log.info(f"Warning: Actual batch size ({len(fname_list)}) is less than 50.")
    if dry_run:
        log.info("Dry run!")
        sys.exit()
    # load TPFs
    tpfs = get_tpfs(fname_list, tar_tpfs=tar_tpfs)
    # create machine object
    machine = pm.TPFMachine.from_TPFs(tpfs, **config)
    del tpfs
    log.info(machine)
    # load shape model from FFI and fit light curves
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
        machine.fit_lightcurves(
            iter_negative=fit_va,
            sap=True,
            fit_va=fit_va,
            load_shape_model=True,
            shape_model_file=shape_model_path,
            plot=False,
        )
    else:
        log.info("No shape model for this Q/Ch, fitting PRF from data...")
        machine.fit_lightcurves(
            iter_negative=fit_va,
            sap=True,
            fit_va=fit_va,
            load_shape_model=False,
            plot=False,
        )
    # compute source centroids
    centroid_method = "scene"
    machine.get_source_centroids(method=centroid_method)
    # save plot if asked
    if plot:
        dir_name = "%s/figures/tpf/ch%02i" % (OUTPUT_PATH, channel)
        log.info(f"Saving diagnostic plots into: {dir_name}")
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        file_name = "%s/%s_models_ch%02i_q%02i_bno%02i_%03i.pdf" % (
            dir_name,
            machine.tpf_meta["mission"][0],
            channel,
            quarter,
            batch_number,
            batch_size,
        )
        shape_fig = machine.plot_shape_model()
        time_fig = machine.plot_time_model()
        with PdfPages(file_name) as pages:
            FigureCanvasPdf(shape_fig).print_figure(pages)
            FigureCanvasPdf(time_fig).print_figure(pages)
        plt.close()

    # get an index array to match the TPF cadenceno
    cadno_mask = np.in1d(machine.tpfs[0].time.jd, machine.time)
    # get KICs
    kics = get_KICs(machine.sources)

    # get the TPF index for each lc, a sources could fall in more than 1 tpf
    tpf_idx = []
    for i in range(len(machine.sources)):
        tpf_idx.append(
            [k for k, ss in enumerate(machine.tpf_meta["sources"]) if i in ss]
        )
    # save lcs
    dir_name = "%s/%s/ch%02i/q%02i" % (
        LCS_PATH,
        machine.tpf_meta["mission"][0].lower(),
        channel,
        quarter,
    )
    log.info(f"Saving light curves into: {dir_name}")
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    if tar_lcs:
        log.info("LCFs will be tar.gz")
        tar = tarfile.open(
            "%s/kbonus-bkgd_ch%02i_q%02i_v%s_lc_b%02i.tar.gz"
            % (dir_name, channel, quarter, lc_version, batch_number),
            mode="w:gz",
        )
    for i, lc in tqdm(
        enumerate(machine.lcs), total=machine.nsources, desc="Saving LCFs"
    ):
        if np.isnan(lc.flux).all() and np.isnan(lc.sap_flux).all():
            continue
        meta = {
            "PIXINAP": machine.aperture_mask[i].sum(),
            "KEPLERID": kics["kic"][i],
            "KEPMAG": kics["kepmag"][i],
            "TPFORG": machine.tpfs[tpf_idx[i][0]].meta["KEPLERID"],
            "TELESCOP": machine.tpfs[0].meta["TELESCOP"],
            "INSTRUME": machine.tpfs[0].meta["INSTRUME"],
            "OBSMODE": machine.tpfs[0].meta["OBSMODE"],
            "OUTPUT": machine.tpfs[0].meta["OUTPUT"],
            "SEASON": machine.tpfs[0].meta["SEASON"],
            "EQUINOX": machine.tpfs[0].meta["EQUINOX"],
            "cadenceno": machine.tpfs[tpf_idx[i][0]].cadenceno[cadno_mask],
            "quality": machine.tpfs[tpf_idx[i][0]].quality[cadno_mask],
            "centroid_col": vars(machine)[
                "source_centroids_column_%s" % (centroid_method)
            ][i],
            "centroid_row": vars(machine)[
                "source_centroids_row_%s" % (centroid_method)
            ][i],
        }
        hdul = make_hdul(lc, machine.sources.loc[i], meta, fit_va=fit_va)

        target_name = (
            "KIC-%s" % (str(meta["KEPLERID"]))
            if meta["KEPLERID"] != ""
            else machine.sources.designation[i].replace(" ", "-")
        )
        fname = "hlsp_kbonus-bkgd_%s-q%02i_v%s_lc.fits" % (
            target_name,
            quarter,
            lc_version,
            # batch_size,
        )
        # i need to tarball all these FITS file to meet inode quota
        if tar_lcs:
            with tempfile.NamedTemporaryFile(mode="wb") as tmp:
                hdul.writeto(tmp)
                tar.add(tmp.name, arcname=fname)
        else:
            hdul.writeto("%s/%s.gz" % (dir_name, fname), overwrite=True, checksum=True)
    if tar_lcs:
        tar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoEncoder")
    parser.add_argument(
        "--quarter",
        dest="quarter",
        type=int,
        default=5,
        help="Quarter number.",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        type=int,
        default=31,
        help="Channel number",
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
        help="Batch number",
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
        "--log", dest="log", default=None, type=int, help="Logging level"
    )
    args = parser.parse_args()
    # set verbose level for logger
    FORMAT = "%(filename)s:%(lineno)s : %(message)s"
    # logging.basicConfig(stream=sys.stdout, level=args.log, format=FORMAT)
    h2 = logging.StreamHandler(sys.stderr)
    h2.setFormatter(logging.Formatter(FORMAT))
    log.addHandler(h2)
    log.setLevel(args.log)
    log.info(vars(args))
    kwargs = vars(args)
    del kwargs["log"]

    do_lcs(**kwargs)

    log.info("Done!")
