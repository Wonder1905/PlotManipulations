# File: parse_paper2fig1_img_to_VQGAN.py
# Created by Juan A. Rodriguez on 12/06/2022
# Goal: This script is intended to access the json files corresponding to the paper2fig dataset (train, val)
# and convert them to the format required by the VQ-GAN, that is, a txt file containing the image path,

import json
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str,default="/workspace/Plots/03_02_2023/",  help="Path to dataset root, containing json files and figures directory")
args = parser.parse_args()
import glob
if __name__ == '__main__':
    path = args.path
    splits = ['train']
    count = 0
    for split in splits:
        for dir in glob.glob(f"{args.path}/{split}/*"):
            for item in tqdm(glob.glob(dir+"/*")):
                path_img = item
                # append to txt file
                with open(path + '/matplots_img_'+split+'.txt', 'a') as f:
                    f.write(path_img + '\n')
                count += 1
    print(f"Stored {count} images in matplots_img_{split}.txt")
