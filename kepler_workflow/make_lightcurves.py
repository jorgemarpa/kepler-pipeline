import os
import argparse
import yaml
import numpy as np
import pandas as pd
import psfmachine as pm
import lightkurve as lk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages

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
    # load shape model from FFI
    shape_model_path = (
        "../data/shape_models/ffi/ch%02i/%s_ffi_shape_model_ch%02i_q%02i.fits"
        % (args.channel, machine.tpf_meta["mission"][0], args.channel, args.quarter)
    )
    ffi.fit_lightcurves(
        iter_negative=True,
        sap=True,
        fit_va=True,
        load_shape_model=True,
        shape_model_file=shape_model_path,
        plot=False,
    )
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

    # save lcs
    for lc in machine.lcs:
        lc.quality = np.zeros_like(lc.flux)
        lc.centroid_col = np.ones_like(lc.flux) * lc.column
        lc.centroid_row = np.ones_like(lc.flux) * lc.row


if __name__ == "__main__":
    main()
