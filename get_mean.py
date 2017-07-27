from cv2 import imread
import glob

def get_dataset_mean(path):

    dataset_mean = 0
    n_files_1 = len(glob.glob(path+'1/*.*'))
    n_files_2 = len(glob.glob(path + '2/*.*'))
    n_files = n_files_1+n_files_2

    labels = ['1', '2']
    for label in labels:
        curr_path = path + label
        for im_file in glob.glob(curr_path):
            im = imread(im_file, -1)
            dataset_mean += (1./n_files)*im.mean()
    return dataset_mean
