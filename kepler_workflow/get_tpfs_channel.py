import os
import tarfile
import argparse
import numpy as np
from tqdm import tqdm

from paths import ARCHIVE_PATH, OUTPUT_PATH, LCS_PATH, PACKAGEDIR
from make_lightcurves import get_file_list


def main(quarter=5, channel=44, batch_number=-1):

    file_list = get_file_list(quarter, channel, batch_number, tar_tpfs=True)

    out_dir = f"{ARCHIVE_PATH}/download/"
    if not os.path.isdir(out_dir):
        os.makedirs(f"{ARCHIVE_PATH}/download/")

    for k, fname in tqdm(enumerate(file_list), total=len(file_list)):
        tarf = f"{fname.split('/')[0]}_{fname.split('/')[1]}.tar"
        tarf = f"{ARCHIVE_PATH}/data/kepler/tpf/{fname.split('/')[0]}/{tarf}"
        tarfile.open(tarf, mode="r").extract(fname, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--bnumber",
        dest="bnumber",
        type=int,
        default=None,
        help="Batch number",
    )
    args = parser.parse_args()
    main(args.quarter, args.channel, args.bnumber)
    print("Done!")
