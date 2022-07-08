import os
from glob import glob
import pandas as pd
import numpy as np

from paths import ARCHIVE_PATH, OUTPUT_PATH, LCS_PATH, PACKAGEDIR


def do_batch_info_tables(tar_archive=True):

    quarters = np.arange(0, 18)
    channels = np.arange(1, 85)

    nbatches = pd.DataFrame(
        np.zeros((quarters.shape[0], channels.shape[0])),
        index=quarters,
        columns=channels,
        dtype=np.int8,
    )

    for q in quarters:
        tar_string = "_tar" if tar_archive else ""
        file_name = f"{OUTPUT_PATH}/support/kepler_tpf_map_q{q:02}{tar_string}_new.csv"

        if not os.path.isfile(file_name):
            print(f"No TPF mapping for quarter {q}")
            continue

        table = pd.read_csv(file_name, index_col=0)

        nbatches.loc[q] = np.array(
            [table.query(f"channel == {ch}").batch.max() for ch in channels]
        )

    nbatches.fillna(0, inplace=True)
    nbatches = nbatches.astype(np.int8)
    print(nbatches)

    nbatches.to_csv(
        f"{OUTPUT_PATH}/support/kepler_quarter_channel_totalbatches_new.csv"
    )

    for q in quarters:

        with open(
            f"{OUTPUT_PATH}/support/kepler_batch_info_quarter{q}_new.dat", "w"
        ) as f:
            index_count = 0
            f.write("#n q ch bt bn\n")
            for ch in channels:
                for k in range(nbatches.loc[q, ch]):
                    f.write(f"{index_count} {q} {ch} {nbatches.loc[q, ch]} {k+1}\n")
                    index_count += 1


if __name__ == "__main__":
    do_batch_info_tables()
