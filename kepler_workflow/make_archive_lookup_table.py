import os
import sys
import glob
import argparse
import fitsio
import tarfile
import tempfile
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from paths import ARCHIVE_PATH, OUTPUT_PATH

log = logging.getLogger(__name__)

qd_map = {
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

        channels, quarters, ras, decs = np.array(
            [
                [
                    fitsio.read_header(f)["CHANNEL"],
                    fitsio.read_header(f)["QUARTER"],
                    fitsio.read_header(f)["RA_OBJ"],
                    fitsio.read_header(f)["DEC_OBJ"],
                ]
                for f in tpfs
            ]
        ).T
    else:
        tarlist = np.sort(glob.glob("%s/%s/%s_*.tar" % (fits_path, folder, folder)))
        log.info(f"Total number of tarballs in {folder}/: {tarlist.shape[0]}")
        if len(tarlist) == 0:
            raise ValueError(f"No TPFs for selected folder {folder}")
        tpfs, channels, quarters, ras, decs = [], [], [], [], []
        with tempfile.TemporaryDirectory(prefix="temp_fits") as tmpdir:
            for tarf in tqdm(tarlist, desc="Reading headers"):
                kic = tarf.split(".")[0].split("_")[-1]
                fname = f"{kic[:4]}/{kic}/kplr{kic}-{qd_map[quarter]}_lpd-targ.fits.gz"
                try:
                    tarfile.open(tarf, mode="r").extract(fname, tmpdir)
                except KeyError:
                    continue
                tpfs.append(fname)
                header = fitsio.read_header(f"{tmpdir}/{fname}")
                channels.append(header["CHANNEL"])
                quarters.append(header["QUARTER"])
                ras.append(header["RA_OBJ"])
                decs.append(header["DEC_OBJ"])

    df = pd.DataFrame(
        [tpfs, quarters, channels, ras, decs],
        index=["file_name", "quarter", "channel", "ra", "dec"],
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
        "--do-batch",
        dest="do_batch",
        action="store_true",
        default=False,
        help="Computen number of batches per channel/quarter.",
    )
    parser.add_argument(
        "--tar-archive",
        dest="tar_archive",
        action="store_true",
        default=False,
        help="Is archive in tarball files.",
    )
    parser.add_argument(
        "--log", dest="log", default=None, type=int, help="Logging level"
    )
    args = parser.parse_args()
    # set verbose level for logger
    FORMAT = "%(filename)s:%(lineno)s : %(message)s"
    logging.basicConfig(stream=sys.stdout, level=args.log, format=FORMAT)

    log.info(vars(args))
    # kwargs = vars(args)

    if args.concat:
        concatenate(args.quarter, tar_archive=args.tar_archive)
    elif args.do_batch:
        how_many_batches(args.quarter, args.batch_size)
    else:
        do_lookup_table(
            folder=args.folder,
            quarter=args.quarter,
            fits_path=args.path,
            tar_archive=args.tar_archive,
        )
    log.info("Done!")
