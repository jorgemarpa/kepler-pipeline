import os
from glob import glob
from paths import PACKAGEDIR


def main():
    info_list = sorted(glob(f"{PACKAGEDIR}/logs/make_lightcurve_*.info"))
    print(f"Total info files: {len(info_list)}")

    batch_idx_fail = []
    for fname in info_list:
        with open(fname, "r") as f:
            lines = f.readlines()
            quarter = int(lines[10].split(":")[-1])
            if lines[-1][-6:-1] == "Done!":
                continue
            else:
                batch_idx_fail.append(int(lines[1].split(":")[-1]))

    with open(
        f"{PACKAGEDIR}/data/support/fail_batch_index_quarter{quarter}.dat", "w"
    ) as f:
        for k in batch_idx_fail:
            f.write(f"{batch_idx_fail[k]}\n")


if __name__ == "__main__":
    main()
