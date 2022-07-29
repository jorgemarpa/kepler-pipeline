import os
from glob import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from termcolor import colored
from give_me_batch_info import main

from paths import ARCHIVE_PATH, OUTPUT_PATH, LCS_PATH, PACKAGEDIR


def check_make_files():
    info_list = sorted(glob(f"{PACKAGEDIR}/logs/make_lightcurve_*.info"))
    print(f"Total info files: {len(info_list)}")

    batch_idx_fail, quarters, channel, batch_number, ntpfs = [], [], [], [], []
    for fname in info_list:
        print(fname)
        with open(fname, "r") as f:
            lines = f.readlines()
            if lines[-1][-6:-1] == "Done!":
                continue
            else:
                try:
                    batch_idx_fail.append(int(lines[3].split(":")[-1]))
                    batch_number.append(int(lines[4].split(":")[-1]))
                    channel.append(int(lines[6].split(":")[-1]))
                    quarters.append(int(lines[13].split(":")[-1]))
                except IndexError:
                    pass
                try:
                    ntpfs.append(int(lines[20].split(" ")[-2]))
                except IndexError:
                    ntpfs.append(0)

    df = pd.DataFrame(
        np.vstack([batch_idx_fail, quarters, channel, batch_number, ntpfs]),
        index=["batch_idx_fail", "quarters", "channel", "batch_number", "ntpfs"],
    ).T

    df.to_csv(f"{PACKAGEDIR}/data/support/fail_batch_info.csv")

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


def check_quarter_archive(
    quarter, suffix="fvaT_bkgT_augT_sgmF_iteT_cbvT", ext="tar.gz", run=False
):

    batch_numer_org = pd.read_csv(
        f"{PACKAGEDIR}/data/support/kepler_quarter_channel_totalbatches_new.csv"
    )
    index_map = pd.read_csv(
        f"{PACKAGEDIR}/data/support/kepler_batch_info_quarter{quarter}_new.dat",
        sep=" ",
        header=0,
    )
    if quarter in [2, 12]:
        suffix = suffix.replace("bkgT", "bkgF")

    channels = np.arange(1, 85)
    missing_idexes = []
    for ch in channels:
        archive_path = sorted(
            glob(f"{LCS_PATH}/kepler/ch{ch:02}/q{quarter:02}/*_lcs_*{suffix}*.{ext}")
        )
        if len(archive_path) > 0:
            total_batches = archive_path[0].split("/")[-1][34:36]
        else:
            total_batches = None

        batches = np.arange(1, batch_numer_org.iloc[quarter, ch] + 1)
        if len(archive_path) > 0:
            batch_done = [int(x.split("_")[5][-2:]) for x in archive_path]
            missing = batches[~np.isin(batches, batch_done)]
        else:
            missing = batches

        index_map_aux = index_map.query(f"q == {quarter} and ch == {ch}")[
            np.isin(batches, missing)
        ]
        # if len(archive_path) > 0:
        missing_idexes.extend(index_map_aux["#n"].values)

        if len(archive_path) == batch_numer_org.iloc[quarter, ch]:
            color = "green"
            txt = "Done"
        elif len(archive_path) == 0:
            color = "red"
            txt = "Fail"
        else:
            color = "yellow"
            txt = "Partial"
        # color = (
        #     "green" if len(archive_path) == batch_numer_org.iloc[quarter, ch] else "red"
        # )
        # if batch_numer_org.iloc[quarter, ch] == 0:
        #     color = "yellow"
        # text = colored(
        #     f"Channel {ch:02} Q {quarter:02} batches {len(archive_path):02} "
        #     f"/ {batch_numer_org.iloc[quarter, ch]:02}",
        #     color=color,
        # )
        text = colored(
            f"{txt} {len(archive_path):02} / {batch_numer_org.iloc[quarter, ch]:02}",
            color=color,
        )
        print(text)
        if run and len(archive_path) == 0 and batch_numer_org.iloc[quarter, ch] > 0:
            main(channel=ch, quarter=quarter, print_info=False, run=run)

    missing_idexes = np.unique(missing_idexes)
    nof = missing_idexes[::25].shape[0]
    np.savetxt(
        f"{PACKAGEDIR}/data/support/fail_batch_index_quarter{quarter}.dat",
        missing_idexes,
        fmt="%i",
    )

    for k, idxs in enumerate(np.array_split(missing_idexes, nof)):
        np.savetxt(
            f"{PACKAGEDIR}/data/support/fail_batch_index_quarter{quarter}_{k+1}.dat",
            idxs,
            fmt="%i",
        )
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
    parser.add_argument(
        "--run",
        dest="run",
        action="store_true",
        default=False,
        help="Execute PBS job.",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    if args.mode == "make_files":
        check_make_files()
    elif args.mode == "check_archive":
        if args.channel is not None:
            check_channel_archive(args.channel, ext=args.ext)
        if args.quarter is not None:
            check_quarter_archive(args.quarter, ext=args.ext, run=args.run)
