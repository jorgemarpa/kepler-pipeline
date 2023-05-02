import os
import socket

PACKAGEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

if socket.gethostname() in ["NASAs-MacBook-Pro.local"]:
    # ARCHIVE_PATH = "/Users/jorgemarpa/Work/BAERI/ADAP"
    ARCHIVE_PATH = "/Volumes/ADAP-Kepler/Work/BAERI"
    OUTPUT_PATH = f"{PACKAGEDIR}/data"
    LCS_PATH = f"{PACKAGEDIR}/data/lcs"
    # KBONUS_LCS_PATH = "/Volumes/ADAP-KBonus-1/work/kbonus/lcs"
    # KBONUS_LCS_PATH = "/Volumes/ADAP-KBonus-2/work/kbonus/lcs"
    # KBONUS_LCS_PATH = "/Volumes/ADAP-Kepler/Work/BAERI/kbonus/lcs"
    KBONUS_LCS_PATH = (
        "/Users/jorgemarpa/Work/BAERI/ADAP/kepler-workflow/data/lcs/rename"
    )
    # KBONUS_LCS_PATH = f"{PACKAGEDIR}/data/lcs/"
    KBONUS_CAT_PATH = f"{PACKAGEDIR}/data/catalogs"

else:
    ARCHIVE_PATH = "/nobackup/jimartin/ADAP"
    OUTPUT_PATH = f"{PACKAGEDIR}/data"
    LCS_PATH = "/nobackup/jimartin/ADAP/kbonus/lcs"
    KBONUS_CAT_PATH = "/nobackup/jimartin/ADAP/kbonus/catalogs"
    KBONUS_LCS_PATH = "/nobackup/jimartin/ADAP/kbonus/lcs"
