import os
import png
from cv2 import imread, resize
from tqdm import tqdm

data_path = 'data/PA_512_16/M_Adult/'
resize_path = 'data/PA_299_16/M_Adult/'

dirs = ['train', 'test']
labels = ['1', '2']

for dir in dirs:
    for label in labels:
        relevant_data_path = data_path + dir + '/' + label +'/'
        relevant_resize_path = resize_path + dir + '/' + label + '/'
        os.makedirs(relevant_resize_path, exist_ok=True)
        files = os.listdir(relevant_data_path)
        for file in tqdm(files):
            im = imread(relevant_data_path + file, -1)
            im = resize(im, (299, 299))
            with open(relevant_resize_path + file, 'wb') as f:
                writer = png.Writer(width=im.shape[1], height=im.shape[0], bitdepth=16)
                im2list = im.reshape(-1, im.shape[1] * im.shape[2]).tolist()
                writer.write(f, im2list)

