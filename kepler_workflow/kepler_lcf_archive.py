import os
import sys
import glob
import shutil
import tarfile
import argparse
import fitsio
import shutil
import tempfile
from tqdm import tqdm
import numpy as np

from paths import *
KEP_LC_PATH = "/Volumes/Jorge MarPa/Work/BAERI/data/kepler/lcs"


def do_archive(tar_path):

    print(f"Working with file {tar_path}")
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(tar_path) as tar:
            members = tar.getmembers()

            ids = [
                x.name.split("/")[-1].split("-")[0][4:] for x in members
            ]
            dirs = list(set([x[:4] for x in ids]))
            print("Creating the following directories:")
            print(dirs)

            for dir in dirs:
                dir_path = f"{KEP_LC_PATH}/{dir}"
                if not os.path.isdir(dir_path):
                    os.makedirs(dir_path)
            # sys.exit()

            for k, member in tqdm(
                enumerate(members),
                total=len(members),
                desc="Extracting FITS into archive",
            ):
                dirout = f"{KEP_LC_PATH}/{ids[k][:4]}/{ids[k]}"
                if not os.path.isdir(dirout):
                    os.makedirs(dirout)
                fout = f"{dirout}/{member.name}"
                tar.extract(member, path=dirout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tar-path",
        dest="tar_path",
        type=str,
        default="/Volumes/jorge-marpa/Work/BAERI/data/kepler/lcs/quarters/public_Q8_long_1.tgz",
        help="Tarfile name",
    )
    args = parser.parse_args()
    do_archive(args.tar_path)
