import os
import sys
import glob
import argparse
import tarfile
import tempfile
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits

from paths import ARCHIVE_PATH, OUTPUT_PATH

log = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None

qd_map = {
    0: 2009131105131,
    1: 2009166043257,
    2: 2009259160929,
    3: 2009350155506,
    4: 2010078095331,
    5: 2010174085026,
    6: 2010265121752,
    7: 2010355172524,
    8: 2011073133259,
    9: 2011177032512,
    10: 2011271113734,
    11: 2012004120508,
    12: 2012088054726,
    13: 2012179063303,
    14: 2012277125453,
    15: 2013011073258,
    16: 2013098041711,
    17: 2013131215648,
}


def do_lookup_table(
    folder="0007",
    quarter=5,
    fits_path=f"{ARCHIVE_PATH}/data/kepler/tpf",
    tar_archive=True,
    quiet=False,
):

    if not tar_archive:
        tpfs = np.sort(
            glob.glob(
                "%s/%s/*/kplr*-%s_lpd-targ.fits.gz"
                % (fits_path, folder, str(qd_map[quarter]))
            )
        )
        log.info(f"Total number of TPFs in {folder}: {tpfs.shape[0]}")
        if len(tpfs) == 0:
            raise ValueError("No TPFs for selected quarter %i" % quarter)

        channels, quarters, ras, decs, cols, rows = np.array(
            [
                [
                    fits.getheader(f, ext=0)["CHANNEL"],
                    fits.getheader(f, ext=0)["QUARTER"],
                    fits.getheader(f, ext=0)["RA_OBJ"],
                    fits.getheader(f, ext=0)["DEC_OBJ"],
                    fits.getheader(f, ext=1)["1CRV5P"],
                    fits.getheader(f, ext=1)["2CRV5P"],
                ]
                for f in tpfs
            ]
        ).T
    else:
        tarlist = np.sort(glob.glob("%s/%s/%s_*.tar" % (fits_path, folder, folder)))
        log.info(f"Total number of tarballs in {folder}/: {tarlist.shape[0]}")
        if len(tarlist) == 0:
            raise ValueError(f"No TPFs for selected folder {folder}")
        tpfs, channels, quarters, ras, decs, cols, rows = [], [], [], [], [], [], []
        with tempfile.TemporaryDirectory(prefix="temp_fits") as tmpdir:
            for tarf in tqdm(tarlist, desc="Reading headers", disable=quiet):
                kic = tarf.split(".")[0].split("_")[-1]
                fname = f"{kic[:4]}/{kic}/kplr{kic}-{qd_map[quarter]}_lpd-targ.fits.gz"
                try:
                    tarfile.open(tarf, mode="r").extract(fname, tmpdir)
                except KeyError:
                    continue
                except tarfile.ReadError:
                    log.info(f"tar file fail {tarf}")
                    continue
                tpfs.append(fname)
                header = fits.getheader(f"{tmpdir}/{fname}", ext=0)
                channels.append(header["CHANNEL"])
                quarters.append(header["QUARTER"])
                ras.append(header["RA_OBJ"])
                decs.append(header["DEC_OBJ"])
                header = fits.getheader(f"{tmpdir}/{fname}", ext=1)
                cols.append(header["1CRV5P"])
                rows.append(header["2CRV5P"])

    df = pd.DataFrame(
        [tpfs, quarters, channels, ras, decs, cols, rows],
        index=["file_name", "quarter", "channel", "ra", "dec", "col", "row"],
    ).T
    df.channel = df.channel.astype(np.int8)
    df.quarter = df.quarter.astype(np.int8)

    dir_name = f"{OUTPUT_PATH}/support/"
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    file_name = "%s/kepler_tpf_map_%s_q%02i%s.csv" % (
        dir_name,
        folder,
        quarter,
        "_tar" if tar_archive else "",
    )
    df.to_csv(file_name)


def concatenate(quarter, tar_archive=True):

    log.info("Concatenating all lookup tables...")
    f_list = np.sort(
        glob.glob(
            "%s/support/kepler_tpf_map_*_q%02i%s.csv"
            % (OUTPUT_PATH, quarter, "_tar" if tar_archive else "")
        )
    )
    if len(f_list) == 0:
        raise FileExistsError("No files to concatenate")
    dfs = pd.concat([pd.read_csv(f, index_col=0) for f in f_list], axis=0)

    file_name = "%s/support/kepler_tpf_map_q%02i%s.csv" % (
        OUTPUT_PATH,
        quarter,
        "_tar" if tar_archive else "",
    )
    log.info(f"Output file: {file_name}")
    dfs.reset_index(drop=True).to_csv(file_name)
    for f in f_list:
        os.remove(f)


def sort_tpfs_in_all_channel(quarter, tar_archive=True, ncols_start=4):

    file_name = "%s/support/kepler_tpf_map_q%02i%s.csv" % (
        OUTPUT_PATH,
        quarter,
        "_tar" if tar_archive else "",
    )
    lkp_tbl = pd.read_csv(file_name, index_col=0)

    bins = [5, 4, 3, 2, 1]
    sorted_lkp_tbl = []
    log.info(f"Working with Quarter {quarter}")
    for ch in tqdm(range(1, 85), total=84, disable=False):
        files_in = lkp_tbl.query("channel == %i and quarter == %i" % (ch, quarter))
        if len(files_in) == 0:
            continue
        log.info(f"Channel {ch} total TPFS {len(files_in)}")
        if len(files_in) < 1500:
            ncols = ncols_start - 1
        else:
            ncols = ncols_start
        log.info(f"Ncols {ncols}")
        bn = ncols
        sorted_ch = []
        col_size = 1112 // bn
        row_size = 1044 // bn
        bn_row_org = np.arange(bn)
        bn_col = np.arange(bn)
        for i, x in enumerate(range(bn)):
            if i % 2 == 1:
                bn_row = bn_row_org[::-1]
            else:
                bn_row = bn_row_org
            for y in range(bn):

                in_cell = files_in.query(
                    f"col >= {bn_col[x]*col_size} and col <= {(bn_col[x]+1)*col_size} and "
                    f"row >= {bn_row[y]*row_size} and row <= {(bn_row[y]+1)*row_size}"
                )
                sorted_ch.append(in_cell.sort_values(["row"], ascending=i % 2 == 0))

        sorted_ch = pd.concat(sorted_ch).reset_index(drop=True).drop_duplicates()

        df_with_batch = sort_tpfs_in_channel(sorted_ch, ncols=ncols, batch_size=200)
        sorted_lkp_tbl.append(df_with_batch)
        log.info("####" * 10)

    sort_tpfs_in_all_channel = (
        pd.concat(sorted_lkp_tbl).reset_index(drop=True).drop_duplicates()
    )
    if sort_tpfs_in_all_channel.shape[0] != lkp_tbl.shape[0]:
        raise RuntimeError("Missing TPFs")
    sort_tpfs_in_all_channel.to_csv(file_name.replace(".csv", "_new.csv"))

    return


def do_batches_in_col(df, batch_size=200, tolerance=0.5):

    left = len(df) % batch_size

    if left / batch_size < 0.1:
        pass
    elif left / batch_size < tolerance:
        while (len(df) % batch_size) / batch_size > 0.1:
            batch_size += 1
    elif left / batch_size > tolerance:
        while (len(df) % batch_size) / batch_size > 0.1 and batch_size > 170:
            batch_size -= 1
    tot_b = len(df) // batch_size

    log.info(batch_size, tot_b)
    aux = np.zeros(len(df))
    batch_index = np.hstack([np.ones(batch_size) * (k + 1) for k in range(tot_b)])
    aux[: len(batch_index)] = batch_index
    aux[aux == 0] = np.max(batch_index)
    df.loc[:, "batch"] = aux

    return df


def sort_tpfs_in_channel(df, ncols=4, batch_size=200):
    col_lims = np.linspace(0, 1112, ncols + 1)
    sort_new = []
    prev_batch = 0
    for x in range(len(col_lims) - 1):
        in_col = df.query(f"col >= {col_lims[x]} and col < {col_lims[x + 1]}")
        in_col_sorted = do_batches_in_col(in_col, batch_size=batch_size)
        aux = in_col_sorted["batch"].max()
        in_col_sorted.loc[:, "batch"] += prev_batch
        sort_new.append(in_col_sorted)

        prev_batch += aux

    return pd.concat(sort_new, axis=0).reset_index(drop=True)


def how_many_batches(quarter, batch_size):
    file_name = "%s/support/kepler_tpf_map_all_q%02i.csv" % (OUTPUT_PATH, quarter)
    df = pd.read_csv(file_name, index_col=0)

    channels = np.arange(1, 85)
    number_batch, nsources = [], []
    for ch in channels:
        in_channel = df.query("channel == %i" % ch)
        nsources.append(in_channel.shape[0])
        number_batch.append(int(np.ceil(in_channel.shape[0] / batch_size)))
    df_nb = pd.DataFrame(
        np.vstack([channels, nsources, number_batch]).T,
        columns=["channel", "n_sources", "n_batch"],
    )
    # log.info(df_nb.set_index("channel"))

    file_name = "%s/support/kepler_tpf_nbatches_bs%03i_q%02i.csv" % (
        OUTPUT_PATH,
        batch_size,
        quarter,
    )
    df_nb.set_index("channel").to_csv(file_name)


def how_many_tpfs(tar_archive=True):
    df = pd.DataFrame(
        np.zeros((18, 84), dtype=int), index=np.arange(0, 18), columns=np.arange(1, 85)
    )
    for q in df.index:
        file_name = "%s/support/kepler_tpf_map_q%02i%s.csv" % (
            OUTPUT_PATH,
            q,
            "_tar" if tar_archive else "",
        )
        if not os.path.isfile(file_name):
            log.info(f"Warning: no file map for quarter {q}")
            continue
        map = pd.read_csv(file_name, index_col=0)
        for ch in df.columns:
            df.loc[q, ch] = map.query(f"channel == {ch}").shape[0]

    file_name = "%s/support/kepler_ntpf_qch.csv" % (OUTPUT_PATH)
    df.to_csv(file_name)


# def do_tpf_batch_files():
#     file_name = "../data/support/kepler_tpf_map_all_q%02i.csv" % (args.quarter)
#     df = pd.read_csv(file_name, index_col=0)
#
#     channels = np.arange(1, 85)
#     channels_batch_dict = {}
#     for k, ch in enumerate(channels):
#         in_channel = df.query("channel == %i" % ch)
#         if files_in.shape[0] == 0:
#             log.info(f"Channel {ch} does not contain TPFs.")
#             continue
#
#         n_batches = int(np.ceil(in_channel.shape[0] / args.batch_size))
#         batch_dict = {}
#         for nb in range(1, n_batches + 1):
#             files_in_batch = in_channel.iloc[
#                 batch_size * (batch_number - 1) : batch_size * (batch_number)
#             ]
#             if files_in_batch.shape[0] < batch_size / 4:
#                 batch_dict[nb - 1].extend(files_in_batch.file_name.tolist())
#             else:
#                 batch_dict[nb] = files_in_batch.file_name.tolist()
#         channels_batch_dict[ch] = batch_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create lookup tables with FITS file path and creates batches"
    )
    parser.add_argument(
        "--folder",
        dest="folder",
        type=str,
        default="0007",
        help="First level folder name of Kepler archive directory.",
    )
    parser.add_argument(
        "--quarter",
        dest="quarter",
        type=int,
        default=5,
        help="First level folder name of Kepler archive directory.",
    )
    parser.add_argument(
        "--path",
        dest="path",
        type=str,
        default="/Users/jorgemarpa/Work/BAERI/ADAP/data/kepler/tpf",
        help="Kepler archive path.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=200,
        help="Batch size",
    )
    parser.add_argument(
        "--concat",
        dest="concat",
        action="store_true",
        default=False,
        help="Concatenate all lookup tables in a quarter.",
    )
    parser.add_argument(
        "--sort",
        dest="sort",
        action="store_true",
        default=False,
        help="Sort TPFs.",
    )
    parser.add_argument(
        "--sum-tpfs",
        dest="sum_tpfs",
        action="store_true",
        default=False,
        help="Computen number of batches per channel/quarter.",
    )
    parser.add_argument(
        "--tar-tpfs",
        dest="tar_archive",
        action="store_true",
        default=False,
        help="Is archive in tarball files.",
    )
    parser.add_argument("--log", dest="log", default=0, help="Logging level")
    args = parser.parse_args()
    # set verbose level for logger
    try:
        args.log = int(args.log)
    except:
        args.log = str(args.log.upper())
    FORMAT = "%(filename)s:%(lineno)s : %(message)s"
    h2 = logging.StreamHandler(sys.stderr)
    h2.setFormatter(logging.Formatter(FORMAT))
    log.addHandler(h2)
    log.setLevel(args.log)
    log.info(vars(args))

    if args.concat:
        concatenate(args.quarter, tar_archive=args.tar_archive)
        sort_tpfs_in_all_channel(
            args.quarter, tar_archive=args.tar_archive, ncols_start=4
        )
    elif args.sort:
        sort_tpfs_in_all_channel(
            args.quarter, tar_archive=args.tar_archive, ncols_start=4
        )
    elif args.sum_tpfs:
        how_many_tpfs(tar_archive=args.tar_archive)
    else:
        do_lookup_table(
            folder=args.folder,
            quarter=args.quarter,
            fits_path=args.path,
            tar_archive=args.tar_archive,
            quiet=True if args.log in [0, "0", "NOTSET"] else False,
        )
    log.info("Done!")
