import argparse
import glob

from cv2 import imread


def get_dataset_mean(path):
    dataset_mean = 0
    count = 0

    for r, dirs, files in os.walk(path):
        for dr in dirs:
            count += len(glob.glob(os.path.join(r, dr + "/*")))
        for file in files:
            im = imread(os.path.join(r, file), -1)
            dataset_mean += (1. / count) * im.mean()

    # labels = ['1', '2']
    # for label in labels:
    #     curr_path = path + label
    #     for im_file in glob.glob(curr_path):
    #         im = imread(im_file, -1)
    #         dataset_mean += (1./count)*im.mean()
    return dataset_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='data/PA_512_16/M_Adult/train/', help='path to training set')

    args = parser.parse_args()
    ds_mean = get_dataset_mean(args.path)
    print(ds_mean)


if __name__ == '__main__':
    main()
