import os.path
from os.path import join
import shutil
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--src', type=str, default="/home/jing/Documents/work/data/ThreeDoD")
parser.add_argument('--tgt', type=str, default="/home/jing/Documents/work/data/ThreeRGBD")
parser.add_argument('--important_tgt', type=str, default="/home/jing/Documents/work/data/ThreeDoD_im")
args = parser.parse_args()

src_scans = join(args.src, "scans")
tgt_scans = join(args.tgt, "scans")
im_scans = join(args.important_tgt, "scans")

files = os.listdir(src_scans)

for idx in tqdm(range(len(files)), desc="process frames"):
    file = files[idx]
    if os.path.exists(join(src_scans, file, "rgbd")):
        src_folder = join(src_scans, file)
        tgt_folder = join(tgt_scans, file)
        if not os.path.exists(tgt_folder):
            os.makedirs(tgt_folder)
        im_folder = join(im_scans, file)
        if not os.path.exists(im_folder):
            os.makedirs(im_folder)
        shutil.copy(join(src_folder, "frame_info.pkl"), join(tgt_folder, "frame_info.pkl"))
        shutil.copy(join(src_folder, "frame_info.pkl"), join(im_folder, "frame_info.pkl"))
        shutil.copytree(join(src_folder, "rgbd"), join(tgt_folder, "rgbd"))
