import cv2
from tqdm import tqdm
import os
import png
import argparse


def flatten(path, flat_path):
    for root, dirs, files in os.walk(path):
        for name in tqdm(files):
            file_path = os.path.join(root, name)
            flat_root = flat_path + '/'.join(root.split('/')[-2:])
            new_file_path = os.path.join(flat_root, name)
            im = cv2.imread(file_path, -1)
            flat_im = im[:,:,0]
            with open(new_file_path, 'wb') as f:
                writer = png.Writer(width=flat_im.shape[1], height=flat_im.shape[0], bitdepth=16)
                im_16_2list = im_16.reshape(-1, flat_im.shape[1]).tolist()
                writer.write(f, im_16_2list)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='Path to RGB images')
    parser.add_argument('--flat_path', required=True, help='Path to save grayscale images')

    args = parser.parse_args()
    path = args.path
    flat_path = args.flat_path

    flatten(path, flat_path)


if __name__ == '__main__':
    main()