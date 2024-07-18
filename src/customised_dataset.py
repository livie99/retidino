import os
import shutil
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from contour import contour
import time


### Example for how to use the customized dataset
## When loading training dataset:
## dataset_train = Custom_dataset(transform = transform, data = 'train')
## When loading validation dataset:
## dataset_val = Custom_dataset(transform = transform, data = 'val')

class Custom_dataset():
    """
    A custom dataset class for handling image data.

    Args:
        transform (callable): A function/transform that takes in an image and returns a transformed version.
        data (str): The type of data to load. Can be either 'train' or 'val'.

    Attributes:
        root_data_dir (str): The root directory of the image data.
        transform (callable): The transformation function for the images.
        df_data (pandas.DataFrame): The dataframe containing the image labels and other information.
        df_data_idx (numpy.ndarray): The indexed dataframe values.
        imgs (numpy.ndarray): The array of images.

    Methods:
        __len__(): Returns the length of the dataset.
        get_contours(): Retrieves the contours of the images.
        __getitem__(idx): Retrieves an item from the dataset.

    """

    def __init__(self, transform, data='train'):
        self.root_data_dir = "/home/livieymli/t1riskengine/data/mosaics_graded"
        self.transform = transform

        if data == 'train':
            label_csv = "/home/livieymli/retidino/mosaic/train_s0.csv"
        elif data == 'val':
            label_csv = "/home/livieymli/retidino/mosaic/val_s0.csv"
        else:
            raise Exception("Please specify the data type: 'train' or 'val'")

        self.df_data = pd.read_csv(label_csv)
        self.df_data['id'] = self.df_data.index
        df_data_idx = self.df_data.loc[self.df_data.index.repeat(self.df_data["n_fields"])]
        df_data_idx.reset_index(drop=True, inplace=True)
        self.df_data_idx = df_data_idx.values
        self.imgs = self.get_contours()

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.df_data)

    def get_contours(self):
        """
        Retrieves the contours of the images.

        Returns:
            numpy.ndarray: The array of images.

        """
        comp_imgs = np.empty((0, 384, 384, 3))
        for i in range(len(self.df_data)):
            eg_folder = os.path.join(self.root_data_dir, str(self.df_data["label"][i]))
            imgs = contour(eg_folder, self.df_data["name"][i])

            print(f"Processed {i+1} images", flush=True)

            if imgs is not None and len(imgs) >= 3:
                comp_imgs = np.concatenate((comp_imgs, imgs), axis=0)
            else:
                continue
        return comp_imgs

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the item's id, image tensor, and label.

        """
        img = self.imgs[idx]
        label = self.df_data_idx["label"][idx]
        img_tensor = img
        id = self.df_data_idx["id"][idx]
        return id, img_tensor, label
       


import pandas as pd

def test():
    """
    This function reads a CSV file, duplicates rows based on the 'n_fields' column,
    and prints the resulting DataFrame.
    """
    # Read the CSV file
    df_data = pd.read_csv("/home/livieymli/retidino/mosaic/val_s0.csv")
    
    # Add an 'id' column with the index values
    df_data['id'] = df_data.index
    
    # Duplicate rows based on the 'n_fields' column
    df_data = df_data.loc[df_data.index.repeat(df_data["n_fields"])]
    
    # Reset the index of the DataFrame
    df_data.reset_index(drop=True, inplace=True)
    
    # Print the resulting DataFrame
    print(df_data)

import time
import cv2

def main():
    """
    This function processes a custom dataset and saves cropped images.

    """
    ## test the custom dataset
    start_time = time.time()
    dataset = Custom_dataset(transform=None, data='val')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Started at: {start_time}", flush=True)
    print(f"Ended at: {end_time}", flush=True)
    print(f"Elapsed time: {elapsed_time} seconds", flush=True)
    print(f"Number of images: {len(dataset)}", flush=True)
    
    ## save the cropped images
    for i in range(10):
        eg_id, eg_img, eg_label = dataset[i]
        crop_path = f"/home/livieymli/retidino/ignore/eg_{eg_id}_{i}.jpg"
        cv2.imwrite(crop_path, eg_img)

    
            
if __name__ == "__main__":
    
    main()
    

