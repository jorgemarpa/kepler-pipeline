import argparse
import tarfile
import tempfile
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.time import Time

from psfmachine.utils import get_gaia_sources
from make_lightcurves import get_file_list, get_tpfs
from paths import ARCHIVE_PATH, OUTPUT_PATH, LCS_PATH, PACKAGEDIR

# warnings.filterwarnings("ignore", category=SettingWithCopyWarning)


def _get_coord_and_query_gaia(tpfs, magnitude_limit=18, dr=3, do_query=True):
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
    kics = np.array([x.targetid for x in tpfs])
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
    cone_search = {"ids": kics, "ras": ras, "decs": decs, "rads": rads}

    # query Gaia with epoch propagation
    if do_query:
        sources = get_gaia_sources(
            tuple(ras),
            tuple(decs),
            tuple(rads),
            magnitude_limit=magnitude_limit,
            epoch=Time(tpfs[0].time[len(tpfs[0]) // 2], format="jd").jyear,
            dr=dr,
        )
    else:
        sources = None
    return sources, cone_search


def main(quarter=1, channel=1, do_batch=True, tar_tpfs=False, do_query=False):

    fname_list = get_file_list(quarter, channel, -1, tar_tpfs=tar_tpfs)
    print(f"Total TPF files: {len(fname_list)}")

    if do_batch:
        batch_size = 100
        nbatches = len(fname_list) // batch_size + 1
        tpfs = []
        sources_all, cone_search_all = [], []
        for batch in tqdm(range(nbatches), total=nbatches):
            aux = get_tpfs(
                fname_list[batch_size * (batch) : batch_size * (batch + 1)],
                tar_tpfs=tar_tpfs,
            )
            sources, cone_search = _get_coord_and_query_gaia(
                aux, magnitude_limit=20, dr=3, do_query=do_query
            )
            cone_search = pd.DataFrame.from_dict(cone_search)
            sources_all.append(sources)
            cone_search_all.append(cone_search)

        if do_query:
            sources_all = pd.concat(sources_all, axis=0).reset_index(drop=True)
        cone_search_all = pd.concat(cone_search_all, axis=0).reset_index(drop=True)
    else:
        tpfs = get_tpfs(fname_list, tar_tpfs=tar_tpfs)
        sources_all, cone_search_all = _get_coord_and_query_gaia(
            aux, magnitude_limit=20, dr=3, do_query=do_query
        )
        cone_search_all.append(cone_search_all)

    cone_file = (
        f"{ARCHIVE_PATH}/data/catalogs/"
        f"cone_search_kepler_field_ch{channel:02}_q{quarter:02}.csv"
    )
    cone_search_all.to_csv(cone_file)

    if do_query:
        print(f"Total sources {len(sources_all)}")
        file_out = (
            f"{ARCHIVE_PATH}/data/catalogs/"
            f"kic_x_gaia_edr3_kepler_field_q{quarter:02}_ch{channel:02}.csv"
        )
        sources_all.to_csv(file_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoEncoder")
    parser.add_argument(
        "--channel",
        dest="channel",
        type=int,
        default=31,
        help="Channel number",
    )
    parser.add_argument(
        "--quarter",
        dest="quarter",
        type=int,
        default=1,
        help="Channel number",
    )
    parser.add_argument(
        "--tar-tpfs",
        dest="tar_tpfs",
        action="store_true",
        default=False,
        help="Is archive in tarball files.",
    )
    args = parser.parse_args()
    print(f"Channel {args.channel}")
    main(quarter=args.quarter, channel=args.channel, tar_tpfs=args.tar_tpfs)
