import argparse
import pandas as pd
import os
from tqdm import tqdm
from shutil import copyfile
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data_path', type=str,
                        help="path to the raw data as it is on Kaggle")
    parser.add_argument('output_folder_path', type=str,
                        help="path to the folder that will contain the newly organized data")
    parser.add_argument('--test_size', type=int, default=0.2,
                        help="the test data ratio")
    return parser.parse_args()


def organize_data(raw_data_path, output_folder_path, test_size):
    train_dir = output_folder_path + '/train'
    valid_dir = output_folder_path + '/valid'
    train_labels_path = raw_data_path + '/train.csv'

    train_labels = pd.read_csv(train_labels_path)
    x = train_labels['id_code']
    y = train_labels['diagnosis']
    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=test_size,
                                                          stratify=y, random_state=0)
    labels = train_y.unique()

    for label in labels:
        os.makedirs(train_dir + f'/{label}')
        os.makedirs(valid_dir + f'/{label}')

    for x, y, folder in [(train_x, train_y, 'train'), (valid_x, valid_y, 'valid')]:
        for i in tqdm(range(len(x))):
            img_name = x.iloc[i] + '.png'
            label = y.iloc[i]
            img_path = os.path.join('train_images', img_name)
            new_img_path = os.path.join(f'{output_folder_path}/{folder}/{label}', img_name)
            copyfile(img_path, new_img_path)


if __name__ == '__main__':
    args = get_args()
    raw_data_path = args.raw_data_path
    output_folder_path = args.output_folder_path
    test_size = args.test_size
    organize_data(raw_data_path, output_folder_path, test_size)

