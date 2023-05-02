import os, glob, sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from tqdm.auto import tqdm
import argparse

sys.path.append("/Users/jorgemarpa/Work/BAERI/ADAP/kepler-workflow/kepler_workflow")
from paths import *

import warnings

warnings.filterwarnings("ignore", category=u.UnitsWarning)
warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)


channel_map = {
    "2.1": 1,
    "2.2": 2,
    "2.3": 3,
    "2.4": 4,
    "3.1": 5,
    "3.2": 6,
    "3.3": 7,
    "3.4": 8,
    "4.1": 9,
    "4.2": 10,
    "4.3": 11,
    "4.4": 12,
    "6.1": 13,
    "6.2": 14,
    "6.3": 15,
    "6.4": 16,
    "7.1": 17,
    "7.2": 18,
    "7.3": 19,
    "7.4": 20,
    "8.1": 21,
    "8.2": 22,
    "8.3": 23,
    "8.4": 24,
    "9.1": 25,
    "9.2": 26,
    "9.3": 27,
    "9.4": 28,
    "10.1": 29,
    "10.2": 30,
    "10.3": 31,
    "10.4": 32,
    "11.1": 33,
    "11.2": 34,
    "11.3": 35,
    "11.4": 36,
    "12.1": 37,
    "12.2": 38,
    "12.3": 39,
    "12.4": 40,
    "13.1": 41,
    "13.2": 42,
    "13.3": 43,
    "13.4": 44,
    "14.1": 45,
    "14.2": 46,
    "14.3": 47,
    "14.4": 48,
    "15.1": 49,
    "15.2": 50,
    "15.3": 51,
    "15.4": 52,
    "16.1": 53,
    "16.2": 54,
    "16.3": 55,
    "16.4": 56,
    "17.1": 57,
    "17.2": 58,
    "17.3": 59,
    "17.4": 60,
    "18.1": 61,
    "18.2": 62,
    "18.3": 63,
    "18.4": 64,
    "19.1": 65,
    "19.2": 66,
    "19.3": 67,
    "19.4": 68,
    "20.1": 69,
    "20.2": 70,
    "20.3": 71,
    "20.4": 72,
    "22.1": 73,
    "22.2": 74,
    "22.3": 75,
    "22.4": 76,
    "23.1": 77,
    "23.2": 78,
    "23.3": 79,
    "23.4": 80,
    "24.1": 81,
    "24.2": 82,
    "24.3": 83,
    "24.4": 84,
}

MJDi_q0 = 54953.03907252848
MJDf_q17 = 56423.50139254052


def run_code(dirname="0007", verbose=True):

    print(f"{KBONUS_LCS_PATH}/{dirname}/*/*fits")
    files_in_dir = glob.glob(f"{KBONUS_LCS_PATH}/{dirname}/*/*fits")
    if len(files_in_dir) == 0:
        sys.exit(1)

    for f in tqdm(files_in_dir):
        try:
            hdul = fits.open(f)
        except:
            print(f"file: {f}")
            print("Failed loading FITS file, empty or corrupt")
            sys.exit(1)

        hdul[0].header.set(
            "DOI",
            "10.17909/7jbr-w430",
            "Digital Object Identifier for the HLSP data collection",
            before="ORIGIN",
        )
        hdul[0].header.set("HLSPID", "KBONUS-BKG", after="DOI")
        hdul[0].header.set("HLSPLEAD", "Jorge martinez-Palomera", after="HLSPID")
        hdul[0].header.set("HLSPVER", "V1.0", after="HLSPLEAD")
        hdul[0].header.set("LICENSE", "CC BY 4.0", after="HLSPVER")
        hdul[0].header.set(
            "LICENURL", "https://creativecommons.org/licenses/by/4.0/", after="LICENSE"
        )

        hdul[0].header.set("FILTER", "KEPLER", "", after="INSTRUME")
        hdul[0].header.set("TIMESYS", "TDB", "Time scale", after="FILTER")

        hdul[0].header.set("EQUINOX", 2016.0, "Coordinate equinox", after="DEC_OBJ")

        hdul[0].header.set(
            "OBJECT",
            hdul[0].header["LABEL"],
            "String version of Target id",
            before="KEPLERID",
        )
        hdul[0].header.set(
            "TARGETID",
            hdul[0].header["LABEL"].split(" ")[-1],
            "Target identifier",
            after="LABEL",
        )
        hdul[0].header.set(
            "GAIAID", hdul[0].header["GAIAID"], "Gaia designation", before="PMRA"
        )

        hdul[0].header.set(
            "XPOSURE", 29.4244 * 60, "Exposure time [s]", after="OBSMODE"
        )
        hdul[0].header.set(
            "MJD-BEG", MJDi_q0, "Begining of observation", after="XPOSURE"
        )
        hdul[0].header.set("MJD-END", MJDf_q17, "End of observation", after="MJD-BEG")

        del hdul[0].header["CHANNEL"]

        # update time unots to bkjd
        hdul[1].header["TUNIT1"] = "BJD - 2454833"
        lc_ext = [k for k in range(len(hdul)) if "LIGHTCURVE" in hdul[k].name]
        for k in lc_ext[1:]:
            hdul[k].header["TUNIT2"] = "BJD - 2454833"
            hdul[k].header.set("TUNIT12", "e-/s", after="TFORM10")
            hdul[k].header.set(
                "CHANNEL",
                channel_map[f"{hdul[k].header['MODULE']}.{hdul[k].header['OUTPUT']}"],
                "CCD channel number",
                after="OUTPUT",
            )

        hdul.writeto(f, overwrite=True, checksum=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HLSP maker")
    parser.add_argument(
        "--dirname",
        dest="dirname",
        type=str,
        default="0007",
        help="Directory name.",
    )
    args = parser.parse_args()

    print(f"Directory name {args.dirname}")
    run_code(dirname=args.dirname)
    print("Done!")
