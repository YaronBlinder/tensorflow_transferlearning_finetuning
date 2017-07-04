import os
import argparse

import numpy as np
from scipy.misc import imread, imsave
from tqdm import tqdm


def make_rots(im):
    im_90 = np.rot90(im, 1)
    im_180 = np.rot90(im, 2)
    im_270 = np.rot90(im, 3)
    return im_90, im_180, im_270

def augment_ims(path):
    im_list = os.listdir(path)
    for file in tqdm(im_list):
        im = imread(path+file)
        im_90, im_180, im_270 = make_rots(im)
        file_90 = file.split('.')[0]+'_90.'+file.split('.')[1]
        file_180 = file.split('.')[0] + '_180.' + file.split('.')[1]
        file_270 = file.split('.')[0] + '_270.' + file.split('.')[1]
        imsave(path+file_90, im_90)
        imsave(path+file_180, im_180)
        imsave(path+file_270, im_270)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='path to images')


    args = parser.parse_args()
    augment_ims(args.path)


if __name__ == '__main__':
    main()



