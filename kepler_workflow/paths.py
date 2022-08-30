import os
import socket

PACKAGEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

if socket.gethostname() in ["NASAs-MacBook-Pro.local", "NASAs-MacBook-Pro-2.local"]:
    # ARCHIVE_PATH = "/Users/jorgemarpa/Work/BAERI/ADAP"
    ARCHIVE_PATH = "/Volumes/jorge-marpa-personal/work/"
    OUTPUT_PATH = f"{PACKAGEDIR}/data"
    LCS_PATH = f"{PACKAGEDIR}/data/lcs"
    KBONUS_CAT_PATH = f"{PACKAGEDIR}/data/catalogs"

else:
    ARCHIVE_PATH = "/nobackup/jimartin/ADAP"
    OUTPUT_PATH = f"{PACKAGEDIR}/data"
    LCS_PATH = "/nobackup/jimartin/ADAP/kbonus/lcs"
    KBONUS_CAT_PATH = "/nobackup/jimartin/ADAP/kbonus/catalogs"
