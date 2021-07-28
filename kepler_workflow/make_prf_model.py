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
    "--plot",
    dest="plot",
    action="store_true",
    default=False,
    help="Make diagnostic plots.",
)
parser.add_argument(
    "--do_phot",
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

    ffi_files = np.sort(glob.glob("../../data/kepler/ffi/kplr*_ffi-cal.fits"))
    ffi_q_fnames = [
        ffi_f
        for ffi_f in ffi_files
        if fitsio.read_header(ffi_f)["QUARTER"] == args.quarter
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

    print("Building shape model...")
    ax_shape = ffi.build_shape_model(plot=args.plot)
    if args.plot:
        dir_name = "../data/figures/ffi/ch%02i" % args.channel
        print("Saving diagnostic plots into: ", dir_name)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        file_name = "%s/shape_model_ch%02i_q%02i.pdf" % (
            dir_name,
            args.channel,
            args.quarter,
        )
        plt.savefig(file_name, bbox_inches="tight")
        plt.close()

        if args.cut_out:
            ax_img = ffi.plot_image(sources=True)
            file_name = "%s/ffi_image_ch%02i_q%02i.pdf" % (
                dir_name,
                args.channel,
                args.quarter,
            )
            plt.savefig(file_name, bbox_inches="tight")
            plt.close()

    dir_name = "../data/shape_models/ffi/ch%02i" % args.channel
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    file_name = "%s/%s_ffi_shape_model_ch%02i_q%02i.fits" % (
        dir_name,
        ffi.meta["MISSION"],
        args.channel,
        args.quarter,
    )
    ffi.save_shape_model(file_name)
    print("Shape model was saved to: ", file_name)

    if args.do_phot:
        dir_name = "../data/catalogs/ffi/ch%02i" % args.channel
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        file_name = "%s/%s_ffi_catalog_ch%02i_q%02i.fits" % (
            dir_name,
            ffi.meta["MISSION"],
            args.channel,
            args.quarter,
        )
        print("Doing PSF photometry...")
        ffi.save_flux_values(output=file_name)
        print("Catalog was saved to: ", file_name)
    print("Done!")


if __name__ == "__main__":
    main()
