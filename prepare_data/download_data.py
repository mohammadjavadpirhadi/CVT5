from SoccerNet.Downloader import SoccerNetDownloader as SNdl
from getpass import getpass
import os, zipfile, glob, shutil
from tqdm.auto import tqdm

BASE_DIR="/home/sauleh"
LOGS_DIR=f"{BASE_DIR}/logs"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

mySNdl = SNdl(LocalDirectory=f"{BASE_DIR}/data")
mySNdl.password = getpass()
mySNdl.downloadDataTask(
    task="spotting-ball-2024",
    split=["train", "valid", "test", "challenge"],
    password=mySNdl.password
)
mySNdl.downloadGames(
    files=["1_224p.mkv", "2_224p.mkv", "Labels-v2.json", "Labels-caption.json", "Labels-ball.json"],
    split=["train","valid","test","challenge"],
    task="caption"
)

zip_files_dir = glob.glob(f"{BASE_DIR}/data/spotting-ball-2024/*.zip")

for zip_file_dir in tqdm(zip_files_dir, desc="Extracting spotting ball data"):
    with zipfile.ZipFile(zip_file_dir, 'r') as zip_file:
        for file_to_extract in zip_file.namelist():
            if "720p" not in file_to_extract:
                zip_file.extract(file_to_extract, f"{BASE_DIR}/data/")
shutil.rmtree(f"{BASE_DIR}/data/spotting-ball-2024/")
