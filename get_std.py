import argparse
import glob
import os
from tqdm import tqdm
import numpy as np

from cv2 import imread


def get_dataset_std(path, ds_mean):

    count = 0
    acc_var = 0

    for r, dirs, files in os.walk(path):
        for dr in dirs:
            count += len(glob.glob(os.path.join(r, dr + "/*")))
        for file in tqdm(files):
            im = imread(os.path.join(r, file), -1)
            acc_var += (1. / (count-1)) * ((im-ds_mean)**2).sum()

    dataset_std = np.sqrt(acc_var)
    return dataset_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='data/PA_512_16/M_Adult/train/', help='path to training set')
    parser.add_argument('--mean', required=True, help='mean value of dataset')

    args = parser.parse_args()
    dataset_mean = np.float32(args.mean)
    dataset_std = get_dataset_std(args.path, dataset_mean)
    print(dataset_std)


if __name__ == '__main__':
    main()
