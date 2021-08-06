import os
import argparse
import yaml
import numpy as np
import pandas as pd
import psfmachine as pm
import lightkurve as lk
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages

from astropy.coordinates import SkyCoord, match_coordinates_3d
import astropy.units as u
from astroquery.vizier import Vizier

import warnings
from scipy import sparse

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)

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
    default=1,
    help="Channel number",
)
parser.add_argument(
    "--batch-size",
    dest="batch_size",
    type=int,
    default=200,
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
    "--sap",
    dest="sap",
    action="store_true",
    default=False,
    help="Do aperture photometry.",
)
args = parser.parse_args()


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
    catalog.loc[:, "kic"] = None
    catalog.loc[mdist.arcsec < 2, "kic"] = kic[midx[mdist.arcsec < 2]]["KIC"].data.data
    return catalog.kic.values


def get_file_list():

    lookup_table = pd.read_csv("../data/support/kepler_tpf_map_all.csv", index_col=0)
    files_in = lookup_table.query(
        "channel == %i and quarter == %i" % (args.channel, args.quarter)
    )
    files_in = files_in.iloc[
        args.batch_size
        * (args.batch_number - 1) : args.batch_size
        * (args.batch_number)
    ]
    if files_in.shape[0] == 0:
        raise IndexError("Batch does not contain files.")
    return files_in.file_name.tolist()


def make_hdul(lc, catalog, extra_meta):
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
        "CAMPAIGN": lc.meta["CAMPAIGN"],
        # objct info
        "LABEL": lc.meta["LABEL"],
        "TARGETID": lc.meta["TARGETID"],
        "RA_OBJ": lc.meta["RA"],
        "DEC_OBJ": lc.meta["DEC"],
        "EQUINOX": extra_meta["EQUINOX"],
        # KIC info
        "KEPLERID": extra_meta["KEPLERID"],
        # gaia catalog info
        "GAIAID": catalog.designation,
        "PMRA": lc.meta["PMRA"] if np.isfinite(lc.meta["PMRA"]) else None,
        "PMDEC": lc.meta["PMDEC"] if np.isfinite(lc.meta["PMDEC"]) else None,
        "PARALLAX": lc.meta["PARALLAX"] if np.isfinite(lc.meta["PARALLAX"]) else None,
        "GMAG": lc.meta["GMAG"] if np.isfinite(lc.meta["GMAG"]) else None,
        "RPMAG": lc.meta["RPMAG"] if np.isfinite(lc.meta["RPMAG"]) else None,
        "BPMAG": lc.meta["BPMAG"] if np.isfinite(lc.meta["BPMAG"]) else None,
        # extraction info
        "SAP": lc.meta["SAP"],
        "ROW": lc.meta["ROW"],
        "COLUMN": lc.meta["COLUMN"],
        "FLFRCSAP": lc.meta["FLFRCSAP"],
        "CROWDSAP": lc.meta["CROWDSAP"],
    }
    lc_dct = {
        "cadenceno": extra_meta["cadenceno"],
        "time": lc.time,
        "flux": lc.flux,
        "flux_err": lc.flux_err,
        "sap_flux": lc.sap_flux,
        "sap_flux_err": lc.sap_flux_err,
        "psf_flux_nvs": lc.psf_flux_NVA,
        "psf_flux_err_nvs": lc.psf_flux_err_NVA,
        "quality": extra_meta["quality"],
        "centroid_column": extra_meta["centroid_col"],
        "centroid_row": extra_meta["centroid_row"],
    }
    lc_ = lk.LightCurve(lc_dct, meta=meta)
    del lc_dct["time"], lc_dct["flux_err"]
    hdul = lc_.to_fits(**lc_dct, **lc_.meta)
    return hdul


def main():

    # load config file for TPFs
    with open("./tpfmachine_keplerTPFs_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # get TPF file name list
    fname_list = get_file_list()
    # load TPFs
    tpfs = lk.collections.TargetPixelFileCollection([lk.read(f) for f in fname_list])
    # create machine object
    machine = pm.TPFMachine.from_TPFs(tpfs, **config)
    # load shape model from FFI and fit light curves
    shape_model_path = (
        "../data/shape_models/ffi/ch%02i/%s_ffi_shape_model_ch%02i_q%02i.fits"
        % (args.channel, machine.tpf_meta["mission"][0], args.channel, args.quarter)
    )
    machine.fit_lightcurves(
        iter_negative=True,
        sap=True,
        fit_va=True,
        load_shape_model=True,
        shape_model_file=shape_model_path,
        plot=False,
    )
    # compute source centroids
    machine.get_source_centroids()
    # save plot if asked
    if args.plot:
        dir_name = "../data/figures/tpf/ch%02i" % args.channel
        print("Saving diagnostic plots into: ", dir_name)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        file_name = "%s/models_ch%02i_q%02i_bno%02i.pdf" % (
            dir_name,
            args.channel,
            args.quarter,
            args.batch_number,
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
    if True:
        dir_name = "../data/lcs/tpf/ch%02i/q%02i" % (args.channel, args.quarter)
        print("Saving light curves into: ", dir_name)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        for i, lc in tqdm(
            enumerate(machine.lcs), total=machine.nsources, desc="Saving LCFs"
        ):
            if np.isnan(lc.flux).all():
                continue
            meta = {
                "KEPLERID": kics[i],
                "TELESCOP": machine.tpfs[0].meta["TELESCOP"],
                "INSTRUME": machine.tpfs[0].meta["INSTRUME"],
                "OBSMODE": machine.tpfs[0].meta["OBSMODE"],
                "OUTPUT": machine.tpfs[0].meta["OUTPUT"],
                "SEASON": machine.tpfs[0].meta["SEASON"],
                "EQUINOX": machine.tpfs[0].meta["EQUINOX"],
                "cadenceno": machine.tpfs[tpf_idx[i][0]].cadenceno[cadno_mask],
                "quality": machine.tpfs[tpf_idx[i][0]].quality[cadno_mask],
                "centroid_col": machine.source_centroids_column[i],
                "centroid_row": machine.source_centroids_row[i],
            }
            hdul = make_hdul(lc, machine.sources.loc[i], meta)

            fname = "hlsp_kbonus-bkgd_%s-q%02i_v%s_lc.fits.gz" % (
                machine.sources.designation[i].replace(" ", "-"),
                args.quarter,
                "1.0",
            )
            hdul.writeto("%s/%s" % (dir_name, fname), overwrite=True, checksum=True)


if __name__ == "__main__":
    main()
