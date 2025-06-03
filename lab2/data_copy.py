import os
import shutil
import glob

BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
LAB1_TRAIN = os.path.normpath(os.path.join(BASE_DIR, "..", "lab1", "train"))
LAB1_TEST  = os.path.normpath(os.path.join(BASE_DIR, "..", "lab1", "test"))

os.makedirs(RAW_DIR, exist_ok=True)

for src in glob.glob(os.path.join(LAB1_TRAIN, "*.csv")):
    dst = os.path.join(RAW_DIR, os.path.basename(src))
    shutil.copyfile(src, dst)

for src in glob.glob(os.path.join(LAB1_TEST, "*.csv")):
    dst = os.path.join(RAW_DIR, os.path.basename(src))
    shutil.copyfile(src, dst)