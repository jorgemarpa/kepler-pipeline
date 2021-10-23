import argparse
import tarfile
import tempfile
import warnings
import numpy as np
import pandas as pd
from astropy.time import Time

from psfmachine.utils import get_gaia_sources
from psfmachine.tpf import _clean_source_list
from make_lightcurves import get_file_list, get_tpfs
from paths import ARCHIVE_PATH, OUTPUT_PATH, LCS_PATH, PACKAGEDIR

# warnings.filterwarnings("ignore", category=SettingWithCopyWarning)


def _get_coord_and_query_gaia(
    tpfs, magnitude_limit=18, dr=3, ra=None, dec=None, rad=None
):
    """
    Calculate ra, dec coordinates and search radius to query Gaia catalog

    Parameters
    ----------
    tpfs:
    magnitude_limit:
    dr: int
        Which gaia data release to use, default is DR2
    ra : float or list of floats
        RAs to do gaia query
    dec : float or list of floats
        Decs to do gaia query
    rad : float or list of floats
        Radius to do gaia query

    Returns
    -------
    sources: pandas.DataFrame
        Catalog with query result
    """
    # find the max circle per TPF that contain all pixel data to query Gaia
    # CH: Sometimes sources are missing from this...worth checking on
    if (ra is None) & (dec is None) & (rad is None):
        ras1, decs1 = np.asarray(
            [
                tpf.wcs.all_pix2world([np.asarray(tpf.shape[::-1][:2]) + 4], 0)[0]
                for tpf in tpfs
            ]
        ).T
        ras, decs = np.asarray(
            [
                tpf.wcs.all_pix2world([np.asarray(tpf.shape[::-1][:2]) // 2], 0)[0]
                for tpf in tpfs
            ]
        ).T
        rads = np.hypot(ras - ras1, decs - decs1)
    elif (ra is not None) & (dec is not None) & (rad is not None):
        ras, decs, rads = ra, dec, rad
    else:
        raise ValueError("Please set all or None of `ra`, `dec`, `rad`")

    # query Gaia with epoch propagation
    sources = get_gaia_sources(
        tuple(ras),
        tuple(decs),
        tuple(rads),
        magnitude_limit=magnitude_limit,
        epoch=Time(tpfs[0].time[len(tpfs[0]) // 2], format="jd").jyear,
        dr=dr,
    )
    cone_search = {"ras": ras, "decs": decs, "rads": rads}

    ras, decs = [], []
    for tpf in tpfs:
        r, d = np.hstack(tpf.get_coordinates(0)).T.reshape(
            [2, np.product(tpf.shape[1:])]
        )
        ras.append(r)
        decs.append(d)
    ras, decs = np.hstack(ras), np.hstack(decs)
    sources, removed_sources = _clean_source_list(sources, ras, decs)
    return sources, cone_search


def main(channel=1):

    fname_list = get_file_list(5, channel, -1, 1, tar_tpfs=True)
    print(f"Total TPFs: {len(fname_list)}")
    tpfs = get_tpfs(fname_list, tar_tpfs=True)

    sources, cone_search = _get_coord_and_query_gaia(tpfs, magnitude_limit=20, dr=3)

    cone_search = pd.DataFrame.from_dict(cone_search)

    cone_file = (
        f"{ARCHIVE_PATH}/data/catalogs/" f"cone_search_kepler_field_ch{channel:02}.csv"
    )
    cone_search.to_csv(cone_file)

    print(f"Total sources {len(sources)}")
    file_out = (
        f"{ARCHIVE_PATH}/data/catalogs/"
        f"kic_x_gaia_edr3_kepler_field_ch{channel:02}.csv"
    )
    # sources.to_hdf(file_out, key=f"ch{channel:02}", mode="w")
    sources.to_csv(file_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoEncoder")
    parser.add_argument(
        "--channel",
        dest="channel",
        type=int,
        default=31,
        help="Channel number",
    )
    args = parser.parse_args()
    main(channel=args.channel)
