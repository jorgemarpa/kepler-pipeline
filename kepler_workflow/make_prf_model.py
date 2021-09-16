import os
import sys
import glob
import argparse
import fitsio
import numpy as np
import psfmachine as pm
import matplotlib.pyplot as plt

import warnings
from scipy import sparse

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)

parser = argparse.ArgumentParser(description="AutoEncoder")
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
    default=None,
    help="Kepler quarter number.",
)
parser.add_argument(
    "--campaign",
    dest="quarter",
    type=int,
    default=None,
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
    help="Make diagnostic plots.",
)
parser.add_argument(
    "--plot-img",
    dest="plot_img",
    action="store_true",
    default=False,
    help="Make diagnostic plots.",
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
    help="Dry run for testing.",
)
args = parser.parse_args()


def main():

    if args.mission in ["Kepler", "kepler"]:
        ffi_files = np.sort(glob.glob("../../data/kepler/ffi/kplr*_ffi-cal.fits"))
        epoch_kw = "QUARTER"
    elif args.mission in ["ktwo", "K2", "k2"]:
        ffi_files = np.sort(glob.glob("../../data/k2/ffi/ktwo*_ffi-cal.fits"))
        epoch_kw = "CAMPAIGN"
    else:
        raise ValueError("Worng mission name, choose one of [Kepler, K2]")
    ffi_q_fnames = [
        ffi_f
        for ffi_f in ffi_files
        if fitsio.read_header(ffi_f)[epoch_kw] == args.quarter
    ]
    ffi_q_fnames = [f for f in ffi_q_fnames if not "kplr2009170043915" in f]
    print("Using FFI files: ", ffi_q_fnames)

    ffi = pm.FFIMachine.from_file(
        ffi_q_fnames,
        extension=args.channel,
        cutout_size=400 if args.cut_out else None,
        cutout_origin=[300, 300],
        correct_offsets=False,
    )
    print(ffi)

    print("Building shape model...")
    ax_shape = ffi.build_shape_model(plot=args.plot)
    if args.plot:
        dir_name = "../data/figures/ffi/ch%02i" % args.channel
        print("Saving diagnostic plots into: ", dir_name)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        file_name = "%s/%s_shape_model_ch%02i_%s%02i.pdf" % (
            dir_name,
            ffi.meta["MISSION"],
            args.channel,
            epoch_kw.lower()[0],
            args.quarter,
        )
        plt.savefig(file_name, bbox_inches="tight")
        plt.close()

        if args.plot_img:
            ax_img = ffi.plot_image(sources=True)
            file_name = "%s/%s_ffi_image_ch%02i_%s%02i.pdf" % (
                dir_name,
                ffi.meta["MISSION"],
                args.channel,
                epoch_kw.lower()[0],
                args.quarter,
            )
            plt.savefig(file_name, bbox_inches="tight")
            plt.close()

    dir_name = "../data/shape_models/ffi/ch%02i" % args.channel
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    file_name = "%s/%s_ffi_shape_model_ch%02i_%s%02i.fits" % (
        dir_name,
        ffi.meta["MISSION"],
        args.channel,
        epoch_kw.lower()[0],
        args.quarter,
    )
    ffi.save_shape_model(file_name)
    print("Shape model was saved to: ", file_name)

    if args.do_phot:
        dir_name = "../data/catalogs/ffi/ch%02i" % args.channel
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        file_name = "%s/%s_ffi_catalog_ch%02i_%s%02i.fits" % (
            dir_name,
            ffi.meta["MISSION"],
            args.channel,
            epoch_kw.lower()[0],
            args.quarter,
        )
        print("Doing PSF photometry...")
        ffi.save_flux_values(output=file_name)
        print("Catalog was saved to: ", file_name)
    print("Done!")


if __name__ == "__main__":
    main()
