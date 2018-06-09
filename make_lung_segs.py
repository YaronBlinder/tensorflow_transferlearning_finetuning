from lung_seg.segfinal_yaron import *
from glob import glob
import imageio
from tqdm import tqdm

data_folder = '/Radical_data/data/CXR8/all/'

normal_files = glob(data_folder+'1/*')
abnormal_files = glob(data_folder+'2/*')

for i in tqdm(range(10)):
    normal_path = normal_files[i]
    normal_filename = normal_path.split('/')[-1]
    seg_norm_im = seg_image(normal_path, Nmask_path='lung_seg/nmask.png')
    segmented_normal_path = data_folder + 'lung_seg/1/' + normal_filename
    # segmented_normal_path = 'images/segmented/normal/{}'.format(normal_images[i])
    imageio.imwrite(segmented_normal_path, seg_norm_im)

    abnormal_path = abnormal_files[i]
    abnormal_filename = abnormal_path.split('/')[-1]
    seg_abnorm_im = seg_image(abnormal_path, Nmask_path='lung_seg/nmask.png')
    segmented_abnormal_path = data_folder + 'lung_seg/2/' + abnormal_filename
    # segmented_abnormal_path = 'images/segmented/abnormal/{}'.format(abnormal_images[i])
    imageio.imwrite(segmented_abnormal_path, seg_abnorm_im)