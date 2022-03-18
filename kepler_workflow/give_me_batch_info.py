import os
import sys
import argparse
import numpy as np
import pandas as pd

from paths import ARCHIVE_PATH, OUTPUT_PATH, LCS_PATH, PACKAGEDIR


def main(channel=1, quarter=1):

    batch_size = pd.read_csv(
        f"{PACKAGEDIR}/data/support/kepler_quarter_channel_batchsize.csv",
        index_col=0,
    )
    total_batch = pd.read_csv(
        f"{PACKAGEDIR}/data/support/kepler_quarter_channel_totalbatches.csv",
        index_col=0,
    )

    print(f"Channel: {channel}")
    print(f"Quarter: {quarter}")

    print(f"Batch size   : {batch_size.loc[quarter, str(channel)]}")
    print(f"Total batches: {total_batch.loc[quarter, str(channel)]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoEncoder")
    parser.add_argument(
        "--quarter",
        dest="quarter",
        type=int,
        default=None,
        help="Quarter number.",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        type=int,
        default=None,
        help="Channel number",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    main(**kwargs)
