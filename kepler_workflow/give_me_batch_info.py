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

    if quarter == "all":
        lines = []
        for qua in range(0, 18):
            chqbsize = batch_size.loc[qua, str(channel)]
            chqbtot = total_batch.loc[qua, str(channel)]
            print(
                f'qsub -v "qu={qua},ch={channel},bs={chqbsize},bn={chqbtot}" pbs_quarter_channel.sh'
            )
    else:
        print("-------------------------")
        print(f"Channel: {channel}")
        print(f"Quarter: {quarter}")

        chqbsize = batch_size.loc[quarter, str(channel)]
        chqbtot = total_batch.loc[quarter, str(channel)]
        print(f"Batch size   : {chqbsize}")
        print(f"Total batches: {chqbtot}")
        print(
            f'qsub -v "qu={quarter},ch={channel},bs={chqbsize},bn={chqbtot}" pbs_quarter_channel.sh'
        )
        print("-------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoEncoder")
    parser.add_argument(
        "--quarter",
        dest="quarter",
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
