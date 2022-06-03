import sys
import glob
import os
import numpy as np
import pandas as pd
import argparse
import fitsio
import tarfile
import tempfile
from tqdm import tqdm

from paths import LCS_PATH, PACKAGEDIR


def channel_npz(channel=1, quarter=5, suffix="fvaT_bkgT_augT_sgmT_iteT"):

    fpath = (
        f"{LCS_PATH}/kepler/ch{channel:02}/q{quarter:02}/"
        f"kbonus-bkgd_ch{channel:02}_q{quarter:02}_*-*_{suffix}.npz"
    )
    print(fpath)
    file_list = glob.glob(fpath)
    print(len(file_list))

    flux = []
    flux_err = []
    nva_flux = []
    nva_flux_err = []
    sap_flux = []
    sap_flux_err = []
    chi2 = []
    sources = []
    ra, dec = [], []
    for f in file_list:
        npz = np.load(f, allow_pickle=True)
        time = npz["time"]
        print(npz["flux"].shape)
        flux.append(npz["flux"])
        flux_err.append(npz["flux_err"])
        sap_flux.append(npz["sap_flux"])
        sap_flux_err.append(npz["sap_flux_err"])
        nva_flux.append(npz["psfnva_flux"])
        nva_flux_err.append(npz["psfnva_flux_err"])
        chi2.append(npz["chi2"])
        sources.append(npz["sources"])
        ra.append(npz["ra"])
        dec.append(npz["dec"])

    flux = np.hstack(flux)
    flux_err = np.hstack(flux_err)
    nva_flux = np.hstack(nva_flux)
    nva_flux_err = np.hstack(nva_flux_err)
    sap_flux = np.hstack(sap_flux)
    sap_flux_err = np.hstack(sap_flux_err)
    chi2 = np.hstack(chi2)
    sources = np.concatenate(sources)
    ra = np.concatenate(ra)
    dec = np.concatenate(dec)
    print(flux.shape, sources.shape)

    mask1 = np.nanmean(flux, axis=0) >= 0
    mask2 = ~np.isnan(flux).sum(axis=0).astype(bool)
    mask = mask1 & mask2

    flux = flux[:, mask]
    flux_err = flux_err[:, mask]
    nva_flux = nva_flux[:, mask]
    nva_flux_err = nva_flux_err[:, mask]
    sap_flux = sap_flux[:, mask]
    sap_flux_err = sap_flux_err[:, mask]
    chi2 = chi2[:, mask]
    sources = sources[mask]
    ra = ra[mask]
    dec = dec[mask]
    print("Removing nans/neg")
    print(flux.shape, sources.shape)

    _, unique_idx = np.unique(sources, return_index=True)

    flux = flux[:, unique_idx]
    flux_err = flux_err[:, unique_idx]
    nva_flux = nva_flux[:, unique_idx]
    nva_flux_err = nva_flux_err[:, unique_idx]
    sap_flux = sap_flux[:, unique_idx]
    sap_flux_err = sap_flux_err[:, unique_idx]
    chi2 = chi2[:, unique_idx]
    sources = sources[unique_idx]
    ra = ra[unique_idx]
    dec = dec[unique_idx]
    print("Removing duplicated")
    print(flux.shape, sources.shape)

    np.savez(
        (
            f"{LCS_PATH}/kepler/ch{channel:02}/q{quarter:02}/"
            f"kbonus-bkgd_ch{channel:02}_q{quarter:02}_{suffix}.npz"
        ),
        time=time,
        flux=flux,
        flux_err=flux_err,
        sap_flux=sap_flux,
        sap_flux_err=sap_flux_err,
        psfnva_flux=nva_flux,
        psfnva_flux_err=nva_flux_err,
        chi2=chi2,
        sources=sources,
        ra=ra,
        dec=dec,
    )


def channel_feather(
    channel=1,
    quarter=5,
    suffix="fvaT_bkgT_augT_sgmT_iteT",
    version="1.1.1",
    remove=False,
):

    print(
        f"{LCS_PATH}/kepler/ch{channel:02}/q{quarter:02}/"
        f"kbonus-kepler-bkg_ch{channel:02}_q{quarter:02}_"
        f"v{version}_lcs_*_{suffix}.coord.feather"
    )
    bfiles = sorted(
        glob.glob(
            f"{LCS_PATH}/kepler/ch{channel:02}/q{quarter:02}/"
            f"kbonus-kepler-bkg_ch{channel:02}_q{quarter:02}_"
            f"v{version}_lcs_*_{suffix}.coord.feather"
        )
    )
    if len(bfiles) == 0:
        print("No batches for this channel...")
        sys.exit()

    batch_numer_org = pd.read_csv(
        f"{PACKAGEDIR}/data/support/kepler_quarter_channel_totalbatches.csv",
        index_col=0,
    )
    print(f"Total batches: {len(bfiles)} / {batch_numer_org.iloc[quarter, channel]}")
    if len(bfiles) != batch_numer_org.iloc[quarter, channel]:
        print("Channel is uncompleted, concatenation aborted")
        sys.exit()

    (
        coord,
        psf_flux,
        psf_flux_err,
        psfnova_flux,
        psfnova_flux_err,
        sap_flux,
        sap_flux_err,
        chi2,
    ) = ([], [], [], [], [], [], [], [])
    time = pd.read_feather(bfiles[0].replace("coord", "psf")).iloc[:, :2]
    for name in bfiles:
        coord.append(pd.read_feather(name))
        print("Shape of batch:", coord[-1].shape)
        psf_flux.append(pd.read_feather(name.replace("coord", "psf")).iloc[:, 2:].T)
        psf_flux_err.append(
            pd.read_feather(name.replace("coord", "psf_err")).iloc[:, 2:].T
        )
        psfnova_flux.append(
            pd.read_feather(name.replace("coord", "novapsf")).iloc[:, 2:].T
        )
        psfnova_flux_err.append(
            pd.read_feather(name.replace("coord", "novapsf_err")).iloc[:, 2:].T
        )
        sap_flux.append(pd.read_feather(name.replace("coord", "sap")).iloc[:, 2:].T)
        sap_flux_err.append(
            pd.read_feather(name.replace("coord", "sap_err")).iloc[:, 2:].T
        )
        chi2.append(pd.read_feather(name.replace("coord", "chi2")).iloc[:, 2:].T)

    coord = pd.concat(coord).set_index("index")
    psf_flux = pd.concat(psf_flux)
    psf_flux_err = pd.concat(psf_flux_err)
    psfnova_flux = pd.concat(psfnova_flux)
    psfnova_flux_err = pd.concat(psfnova_flux_err)
    sap_flux = pd.concat(sap_flux)
    sap_flux_err = pd.concat(sap_flux_err)
    chi2 = pd.concat(chi2)

    print("Removing nans/neg")
    mask1 = np.mean(psf_flux, axis=1) >= 0
    mask2 = ~np.isnan(psf_flux).sum(axis=1).astype(bool)
    mask = mask1 & mask2

    coord = coord[mask]
    psf_flux = psf_flux[mask]
    psf_flux_err = psf_flux_err[mask]
    psfnova_flux = psfnova_flux[mask]
    psfnova_flux_err = psfnova_flux_err[mask]
    sap_flux = sap_flux[mask]
    sap_flux_err = sap_flux_err[mask]
    chi2 = chi2[mask]

    print("Removing duplicated")
    _, unique_idx = np.unique(coord.index, return_index=True)

    coord = coord.iloc[unique_idx]
    psf_flux = psf_flux.iloc[unique_idx]
    psf_flux_err = psf_flux_err.iloc[unique_idx]
    psfnova_flux = psfnova_flux.iloc[unique_idx]
    psfnova_flux_err = psfnova_flux_err.iloc[unique_idx]
    sap_flux = sap_flux.iloc[unique_idx]
    sap_flux_err = sap_flux_err.iloc[unique_idx]
    chi2 = chi2.iloc[unique_idx]

    outname = (
        bfiles[0].replace(bfiles[0].split()[-1].split("_")[5], "").replace("lcs__", "")
    )
    print(outname)

    print(coord.shape)
    print(time.shape)
    print(psf_flux.shape)
    print(psf_flux_err.shape)
    print(psfnova_flux.shape)
    print(psfnova_flux_err.shape)
    print(sap_flux.shape)
    print(sap_flux_err.shape)
    print(chi2.shape)

    coord.reset_index().to_feather(outname)
    time.to_feather(outname.replace("coord", "time"))
    psf_flux.T.to_feather(outname.replace("coord", "psf"))
    psf_flux_err.T.to_feather(outname.replace("coord", "psf_err"))
    psfnova_flux.T.to_feather(outname.replace("coord", "novapsf"))
    psfnova_flux_err.T.to_feather(outname.replace("coord", "novapsf_err"))
    sap_flux.T.to_feather(outname.replace("coord", "sap"))
    sap_flux_err.T.to_feather(outname.replace("coord", "sap_err"))
    chi2.T.to_feather(outname.replace("coord", "chi2"))

    if remove:
        print("Removing batch files...")
        for f in bfiles:
            os.remove(f)
            os.remove(f.replace("coord", "psf"))
            os.remove(f.replace("coord", "psf_err"))
            os.remove(f.replace("coord", "novapsf"))
            os.remove(f.replace("coord", "novapsf_err"))
            os.remove(f.replace("coord", "sap"))
            os.remove(f.replace("coord", "sap_err"))
            os.remove(f.replace("coord", "chi2"))

    return


def quarter_feather(quarter=5, suffix="fvaT_bkgT_augT_sgmT_iteT", version="1.1.1"):

    channels = np.arange(1, 85, dtype=int)

    coord, channel = [], []
    for ch in tqdm(channels, total=len(channels)):
        name = (
            f"{LCS_PATH}/kepler/ch{ch:02}/q{quarter:02}/"
            f"kbonus-kepler-bkg_ch{ch:02}_q{quarter:02}_v{version}_{suffix}.coord.feather"
        )
        if not os.path.isfile(name):
            continue
        aux = pd.read_feather(name)
        aux.loc[:, "channel"] = [ch] * len(aux)
        coord.append(aux)

    coord = pd.concat(coord).set_index("index")

    print(coord.shape)

    name = (
        f"{LCS_PATH}/kepler/"
        f"kbonus-kepler-bkg_q{quarter:02}_v{version}_{suffix}.coord.feather"
    )
    coord.reset_index().to_feather(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate NPZ files from batches")
    parser.add_argument(
        "--quarter",
        dest="quarter",
        type=int,
        default=5,
        help="Quarter number.",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        # type=int,
        default=None,
        help="Channel number",
    )
    parser.add_argument(
        "--suffix",
        dest="suffix",
        type=str,
        default="fvaT_bkgT_augT_sgmT_iteT",
        help="File prefix",
    )
    parser.add_argument(
        "--file-type",
        dest="file_type",
        type=str,
        default="feather",
        help="File type",
    )
    parser.add_argument(
        "--remove",
        dest="remove",
        action="store_true",
        default=False,
        help="Remove batch files after concat.",
    )

    args = parser.parse_args()
    if args.file_type == "npz":
        channel_npz(channel=args.channel, quarter=args.quarter, suffix=args.suffix)
    elif args.file_type == "feather":
        if args.channel == "all":
            quarter_feather(quarter=args.quarter, suffix=args.suffix)
        else:
            channel_feather(
                channel=int(args.channel),
                quarter=args.quarter,
                suffix=args.suffix,
                remove=args.remove,
            )
    else:
        raise ValueError("Wrong file type...")
