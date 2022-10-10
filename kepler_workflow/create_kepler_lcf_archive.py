import os
import glob
import shutil
import tarfile
from tqdm import tqdm

from paths import ARCHIVE_PATH

def main():

    tar_files = glob.glob(f"{ARCHIVE_PATH}/data/kepler/lcs/q05/public_*.tgz")
    print(tar_files)

    for tf in tar_files[1:]:
        tmp_dir = f"{ARCHIVE_PATH}/data/kepler/lcs/q05/tmp"
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        print(f"Unpacking {tf}")
        with tarfile.open(tf) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, tmp_dir)

        fits_list = glob.glob(f"{ARCHIVE_PATH}/data/kepler/lcs/q05/tmp/kplr*_llc.fits")
        kics = [x.split("/")[-1].split("-")[0][4:] for x in fits_list]
        print(kics[:10])

        print("Creating new directiries...")
        kics_s = list(set([x[:4] for x in kics]))
        print(kics_s)

        for dname in kics_s:
            dir_name = f"{ARCHIVE_PATH}/data/kepler/lcs/q05/{dname}"
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)

        for kic, fits in tqdm(zip(kics, fits_list), total=len(kics), desc="Moving FITS"):
            out_name = f"{ARCHIVE_PATH}/data/kepler/lcs/q05/{kic[:4]}/{fits.split('/')[-1]}"
            shutil.move(fits, out_name)


if __name__ == '__main__':
    main()
