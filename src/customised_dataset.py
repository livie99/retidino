import os
import shutil
import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from contour import contour
import time
import torch
from torchvision import transforms, utils

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


    Methods:
        __len__(): Returns the length of the dataset.
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


    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.df_data)


    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the item's id, image tensor, and label.

        """
        # Retrieve the id, label, image name, and image path for the given index
        id = self.df_data["id"][idx]
        label = self.df_data["label"][idx]
        img_name = self.df_data["name"][idx]
        img_path = os.path.join(self.root_data_dir, str(label), img_name)
        
        # Retrieve the x, y, and r values for the given index
        x, y, r = self.df_data["x"][idx], self.df_data["y"][idx], self.df_data["r"][idx]
        
        # Read the image from the specified path
        image = cv2.imread(img_path)
        
        # Resize the image to a fixed size of 4096x4096
        image = cv2.resize(image, (4096, 4096))
        
        # Create an empty mask of the same size as the image
        mask = np.zeros((4096, 4096), dtype=np.uint8)
        
        # Draw a circle on the original image using the specified x, y, and r values
        cv2.circle(image, (int(x), int(y)), int(r), (0, 0, 0), 2)
        
        # Draw a filled circle on the mask using the specified x, y, and r values
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        
        # Apply the mask to the original image to obtain a masked image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Crop the region of the circle from the masked image
        crop_image = masked_image[int(y-r if y-r>0 else 0):int(y+r if y+r<4096 else 4096), 
                                  int(x-r if x-r>0 else 0):int(x+r if x+r<4096 else 4096)]
        
        # Resize the cropped image to a fixed size of 384x384
        crop_image = cv2.resize(crop_image, (384, 384))
        
        # Convert the cropped image to a PIL image
        PIL_image = Image.fromarray(crop_image.astype('uint8'), 'RGB')
        
        # Apply the specified transformation to the PIL image to obtain the image tensor
        img_tensor = self.transform(PIL_image)
        
        # Return a tuple containing the id, image tensor, and label
        return id, img_tensor, label
       


import pandas as pd

def test():
    """
    This function reads a CSV file, duplicates rows based on the 'n_fields' column,
    and prints the resulting DataFrame.
    """
    # Read the CSV file
    df_data = pd.read_csv("/home/livieymli/retidino/mosaic/val_s0.csv")
    
    # Print the resulting DataFrame
    print(df_data)
    pass


def main():
    """
    This function processes a custom dataset and saves cropped images.

    """
    ## test the custom dataset
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    dataset = Custom_dataset(transform=test_transform, data='val')
    
    
    ## save the cropped images
    for i in range(10):
        eg_id, eg_img, eg_label = dataset[i]
        crop_path = f"/home/livieymli/retidino/ignore/eg_{eg_id}_{i}.jpg"
        # cv2.imwrite(crop_path, eg_img)
        utils.save_image(eg_img, crop_path)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        pin_memory=True,
        drop_last=True,
    )

    start_time = time.time()
    for batch in data_loader:
        # Process the batch
        id, images, labels = batch
        # Do something with the images and labels
        # ...
        pass
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Started at: {start_time}", flush=True)
    print(f"Ended at: {end_time}", flush=True)
    print(f"Elapsed time: {elapsed_time} seconds", flush=True)
    print(f"Number of images: {len(dataset)}", flush=True)
    print(f"Data loaded: there are {len(data_loader)} batches.")
            
if __name__ == "__main__":
    
    main()
    # test()
    

