import os
import sys
import argparse
import numpy as np
import pandas as pd

from paths import ARCHIVE_PATH, OUTPUT_PATH, LCS_PATH, PACKAGEDIR


def main(channel=1, quarter=1, print_info=True, run=False):

    total_batch = pd.read_csv(
        f"{PACKAGEDIR}/data/support/kepler_quarter_channel_totalbatches_new.csv",
        index_col=0,
    )

    if quarter == "all":
        lines = []
        for qua in range(0, 18):
            chqbtot = total_batch.loc[qua, str(channel)]
            print(
                f'qsub -v "qu={qua},ch={channel},bn={chqbtot}" pbs_quarter_channel.sh'
            )
    else:
        chqbtot = total_batch.loc[int(quarter), str(channel)]
        if print_info:
            print("-------------------------")
            print(f"Channel: {channel}")
            print(f"Quarter: {quarter}")
            print(f"Total batches: {chqbtot}")
        if run:
            command = f'qsub -v "qu={quarter},ch={channel},bn={chqbtot}" {PACKAGEDIR}/bash/pbs_quarter_channel.sh'
            print(command)
            os.system(command)


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
    parser.add_argument(
        "--run",
        dest="run",
        action="store_true",
        default=False,
        help="Execute PBS job.",
    )
    parser.add_argument(
        "--print-info",
        dest="print_info",
        action="store_true",
        default=False,
        help="Print info or command.",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    main(**kwargs)
