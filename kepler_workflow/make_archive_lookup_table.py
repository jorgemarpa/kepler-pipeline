import os
import glob
import argparse
import fitsio
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="AutoEncoder")
parser.add_argument(
    "--folder",
    dest="folder",
    type=str,
    default="0007",
    help="First level folder name of Kepler archive directory.",
)
parser.add_argument(
    "--path",
    dest="path",
    type=str,
    default="/Users/jorgemarpa/Work/BAERI/ADAP/data/kepler/tpf/Kepler",
    help="Kepler archive path.",
)
parser.add_argument(
    "--concat",
    dest="concat",
    action="store_true",
    default=False,
    help="Concatenate all files.",
)
args = parser.parse_args()


def main():

    tpfs = np.sort(glob.glob("%s/%s/*/*fits.gz" % (args.path, args.folder)))
    print("Total numebr of TPFs: ", tpfs.shape[0])

    channels, quarters = np.array(
        [
            [fitsio.read_header(f)["CHANNEL"], fitsio.read_header(f)["QUARTER"]]
            for f in tpfs
        ]
    ).T

    df = pd.DataFrame(
        [tpfs, quarters, channels], index=["file_name", "quarter", "channel"]
    ).T

    dir_name = "../data/support/"
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    file_name = "%s/kepler_tpf_map_%s.csv" % (dir_name, args.folder)
    df.to_csv(file_name)


def concatenate():

    print("Concatenating all lookup tables...")
    f_list = np.sort(glob.glob("../data/support/kepler_tpf_map_*.csv"))
    dfs = pd.concat([pd.read_csv(f, index_col=0) for f in f_list], axis=0)

    file_name = "../data/support/kepler_tpf_map_all.csv"
    dfs.to_csv(file_name)


if __name__ == "__main__":
    if args.concat:
        concatenate()
    else:
        main()
    print("Done!")
