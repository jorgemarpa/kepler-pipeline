import glob
import numpy as np
import argparse
import fitsio
import tarfile
import tempfile
from tqdm import tqdm

from paths import LCS_PATH


def main(channel=1, quarter=5, sufix="poscorr_sqrt_tk6_tp100_fvaT_bkgT_augT"):

    fpath = f"{LCS_PATH}/kepler/ch{channel:02}/q{quarter:02}/kbonus-bkgd_ch{channel:02}_q{quarter:02}_*-*_{sufix}.npz"
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
        f"{LCS_PATH}/kepler/ch{channel:02}/q{quarter:02}/kbonus-bkgd_ch{channel:02}_q{quarter:02}_{sufix}.npz",
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
        type=int,
        default=None,
        help="Channel number",
    )
    parser.add_argument(
        "--sufix",
        dest="sufix",
        type=str,
        default="poscorT_sqrt_tk6_tp200",
        help="File prefix",
    )

    args = parser.parse_args()
    main(**vars(args))
