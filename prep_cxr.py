import pandas as pd
from tqdm import tqdm
from shutil import move
import os


path = '/home/ubuntu/data/'
for label in ['1', '2']:
    os.makedirs(path+label, exist_ok=True)
info = pd.read_csv(path+'Data_Entry_2017.csv')
for i, row in tqdm(info.iterrows(), total=info.shape[0]):
    filename = row['Image Index']
    file_path = path + 'images/' + filename
    label = '1' if row['Finding Labels']=='No Finding' else '2'
    new_file_path = path + 'images/' + label + '/cxr8_cd' + filename
    move(file_path, new_file_path)
