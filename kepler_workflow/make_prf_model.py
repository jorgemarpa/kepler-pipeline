import os
import sys
import glob
import argparse
import fitsio
import warnings
import logging
import numpy as np
import psfmachine as pm
import matplotlib.pyplot as plt
from scipy import sparse

from paths import ARCHIVE_PATH, OUTPUT_PATH

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)

log = logging.getLogger(__name__)


def do_FFI(
    mission="kepler",
    quarter=5,
    channel=1,
    plot=False,
    plot_img=False,
    do_phot=False,
    cut_out=False,
):

    if mission in ["Kepler", "kepler"]:
        ffi_files = np.sort(
            glob.glob(f"{ARCHIVE_PATH}/data/kepler/ffi/kplr*_ffi-cal.fits")
        )
        epoch_kw = "QUARTER"
    elif mission in ["ktwo", "K2", "k2"]:
        ffi_files = np.sort(glob.glob(f"{ARCHIVE_PATH}/data/k2/ffi/ktwo*_ffi-cal.fits"))
        epoch_kw = "CAMPAIGN"
    else:
        raise ValueError("Wrong mission name, choose one of [Kepler, K2]")
    ffi_q_fnames = [
        ffi_f for ffi_f in ffi_files if fitsio.read_header(ffi_f)[epoch_kw] == quarter
    ]
    ffi_q_fnames = [f for f in ffi_q_fnames if not "kplr2009170043915" in f]
    log.info(f"Using FFI files: {ffi_q_fnames}")

    ffi = pm.FFIMachine.from_file(
        ffi_q_fnames,
        extension=channel,
        cutout_size=400 if cut_out else None,
        cutout_origin=[300, 300],
        correct_offsets=False,
    )
    log.info(ffi)

    log.info("Building shape model...")
    ax_shape = ffi.build_shape_model(plot=plot)
    if plot:
        dir_name = "%s/figures/ffi/ch%02i" % (OUTPUT_PATH, channel)
        log.info(f"Saving diagnostic plots into: {dir_name}")
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        file_name = "%s/%s_shape_model_ch%02i_%s%02i.pdf" % (
            dir_name,
            ffi.meta["MISSION"],
            channel,
            epoch_kw.lower()[0],
            quarter,
        )
        plt.savefig(file_name, bbox_inches="tight")
        plt.close()

        if plot_img:
            ax_img = ffi.plot_image(sources=True)
            file_name = "%s/%s_ffi_image_ch%02i_%s%02i.pdf" % (
                dir_name,
                ffi.meta["MISSION"],
                channel,
                epoch_kw.lower()[0],
                quarter,
            )
            plt.savefig(file_name, bbox_inches="tight")
            plt.close()

    dir_name = "%s/shape_models/ffi/ch%02i" % (OUTPUT_PATH, channel)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    file_name = "%s/%s_ffi_shape_model_ch%02i_%s%02i.fits" % (
        dir_name,
        ffi.meta["MISSION"],
        channel,
        epoch_kw.lower()[0],
        quarter,
    )
    ffi.save_shape_model(file_name)
    log.info(f"Shape model was saved to: {file_name}")

    if do_phot:
        dir_name = "%s/catalogs/ffi/ch%02i" % (OUTPUT_PATH, channel)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        file_name = "%s/%s_ffi_catalog_ch%02i_%s%02i.fits" % (
            dir_name,
            ffi.meta["MISSION"],
            channel,
            epoch_kw.lower()[0],
            quarter,
        )
        log.info("Doing PSF photometry...")
        ffi.save_flux_values(output=file_name)
        log.info(f"Catalog was saved to: {file_name}")
    log.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Program to compute PRF models from FFI"
    )
    parser.add_argument(
        "--mission",
        dest="mission",
        type=str,
        default="kepler",
        help="Mission Kepler or K2.",
    )
    parser.add_argument(
        "--quarter",
        dest="quarter",
        type=int,
        default=5,
        help="Kepler quarter number.",
    )
    parser.add_argument(
        "--campaign",
        dest="quarter",
        type=int,
        default=5,
        help="K2 campaign number.",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        type=int,
        default=1,
        help="Channel number",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=False,
        help="Make mdoel diagnostic plots.",
    )
    parser.add_argument(
        "--plot-image",
        dest="plot_img",
        action="store_true",
        default=False,
        help="Make FFI plots.",
    )
    parser.add_argument(
        "--do-phot",
        dest="do_phot",
        action="store_true",
        default=False,
        help="Fit PSF photometry and save catalog.",
    )
    parser.add_argument(
        "--cut-out",
        dest="cut_out",
        action="store_true",
        default=False,
        help="Use a cutout of the FFI for testing.",
    )
    parser.add_argument(
        "--log", dest="log", default=None, type=int, help="Logging level"
    )
    args = parser.parse_args()
    # set verbose level for logger
    FORMAT = "%(filename)s:%(lineno)s - %(funcName)10s(): %(message)s"
    logging.basicConfig(stream=sys.stdout, level=args.log, format=FORMAT)
    log.info(vars(args))
    kwargs = vars(args)
    del kwargs["log"]

    do_FFI(**kwargs)
