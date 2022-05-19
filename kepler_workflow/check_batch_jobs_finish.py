import os
from glob import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from termcolor import colored

from paths import ARCHIVE_PATH, OUTPUT_PATH, LCS_PATH, PACKAGEDIR


def check_make_files():
    info_list = sorted(glob(f"{PACKAGEDIR}/logs/make_lightcurve_*.info"))
    print(f"Total info files: {len(info_list)}")

    batch_idx_fail, quarters = [], []
    for fname in info_list:
        print(fname)
        with open(fname, "r") as f:
            lines = f.readlines()
            if lines[-1][-6:-1] == "Done!":
                continue
            else:
                try:
                    batch_idx_fail.append(int(lines[2].split(":")[-1]))
                    quarters.append(int(lines[13].split(":")[-1]))
                except IndexError:
                    continue

    batch_idx_fail = np.array(batch_idx_fail)
    quarters = np.array(quarters)
    for k, q in enumerate(set(quarters)):
        with open(
            f"{PACKAGEDIR}/data/support/fail_batch_index_quarter{q}.dat", "w"
        ) as f:
            for k in np.unique(batch_idx_fail[quarters == q]):
                f.write(f"{k}\n")
    return


def check_channel_archive(channel, suffix="fvaT_bkgT_augT_sgmT_iteT", ext="tar.gz"):

    batch_numer_org = pd.read_csv(
        f"{PACKAGEDIR}/data/support/kepler_quarter_channel_totalbatches.csv"
    )

    quarters = np.arange(0, 18)
    for q in quarters:
        archive_path = sorted(
            glob(f"{LCS_PATH}/kepler/ch{channel:02}/q{q:02}/*_lcs_*{suffix}*.{ext}")
        )
        if len(archive_path) > 0:
            total_batches = archive_path[0].split("/")[-1][34:36]
        else:
            total_batches = None
        color = (
            "green" if len(archive_path) == batch_numer_org.iloc[q, channel] else "red"
        )
        if batch_numer_org.iloc[q, channel] == 0:
            color = "yellow"
        text = colored(
            f"Channel {channel:02} Q {q:02} batches {len(archive_path):02} / {batch_numer_org.iloc[q, channel]:02}",
            color=color,
        )
        print(text)
    return


def check_quarter_archive(quarter, suffix="fvaT_bkgT_augT_sgmT_iteT", ext="tar.gz"):

    batch_numer_org = pd.read_csv(
        f"{PACKAGEDIR}/data/support/kepler_quarter_channel_totalbatches.csv"
    )

    channels = np.arange(1, 85)
    for ch in channels:
        archive_path = sorted(
            glob(f"{LCS_PATH}/kepler/ch{ch:02}/q{quarter:02}/*_lcs_*{suffix}*.{ext}")
        )
        if len(archive_path) > 0:
            total_batches = archive_path[0].split("/")[-1][34:36]
        else:
            total_batches = None

        color = (
            "green" if len(archive_path) == batch_numer_org.iloc[quarter, ch] else "red"
        )
        if batch_numer_org.iloc[quarter, ch] == 0:
            color = "yellow"
        text = colored(
            f"Channel {ch:02} Q {quarter:02} batches {len(archive_path):02} / {batch_numer_org.iloc[quarter, ch]:02}",
            color=color,
        )
        print(text)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoEncoder")
    parser.add_argument(
        "--mode",
        dest="mode",
        default="make_files",
        help="Which type of files to check.",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        type=int,
        default=None,
        help="Channel number",
    )
    parser.add_argument(
        "--quarter",
        dest="quarter",
        type=int,
        default=None,
        help="Quarter",
    )
    parser.add_argument(
        "--ext",
        dest="ext",
        type=str,
        default="tar.gz",
        help="File extension",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    if args.mode == "make_files":
        check_make_files()
    elif args.mode == "check_archive":
        if args.channel is not None:
            check_channel_archive(args.channel, ext=args.ext)
        if args.quarter is not None:
            check_quarter_archive(args.quarter, ext=args.ext)
