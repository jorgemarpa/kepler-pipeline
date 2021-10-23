import glob
import numpy as np
import argparse
from astropy.io import fits
import tarfile
import tempfile
from tqdm import tqdm


def main(prefix="", sufix="", path=""):

    print(f"{path}/{prefix}_b*_{sufix}*")
    file_list = glob.glob(f"{path}/{prefix}_b*_{sufix}*")
    print(len(file_list))

    if "weights" in prefix:
        result = []
        for f in file_list:
            result.append(np.load(f).T)
        result = np.concatenate(result, axis=0)
        print(result.shape)
        np.save(f"{path}/{prefix}_{sufix}.npy", result)

    elif "kbonus-bkgd" in prefix:
        fits_list = []
        with tempfile.TemporaryDirectory(prefix="temp_fits") as tmpdir:
            for tf in tqdm(file_list):
                tar = tarfile.open(tf, mode="r")
                tar.extractall(path=tmpdir)
                fits_list.extend(tar.getnames())
                tar.close()
            tmp_file_list = glob.glob(f"{tmpdir}/*.fits")

            tar_out = tarfile.open(f"{path}/{prefix}_{sufix}.tar.gz", mode="w:gz")
            for i, f in tqdm(enumerate(tmp_file_list), total=len(tmp_file_list)):
                tar_out.add(f, arcname=f.split("/")[-1])
            tar_out.close()

    else:
        raise(ValueError)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoEncoder")
    parser.add_argument(
        "--prefix",
        dest="prefix",
        type=str,
        default="weights_va_ch42_q05",
        help="File prefix",
    )
    parser.add_argument(
        "--sufix",
        dest="sufix",
        type=str,
        default="poscorT_sqrt_tk6_tp200",
        help="File prefix",
    )
    parser.add_argument(
        "--path",
        dest="path",
        type=str,
        default="/Users/jorgemarpa/Work/BAERI/ADAP/kepler-workflow/data/lcs/kepler/ch42/q05",
        help="Kepler archive path.",
    )
    args = parser.parse_args()
    main(**vars(args))
