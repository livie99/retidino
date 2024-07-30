import os 
import cv2
from tqdm import tqdm

dir = "/home/livieymli/retidino/optos"
img_ls = os.listdir(dir)
img_ls.sort()
for name in tqdm(img_ls):
    img_path = os.path.join(dir, name)
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"None image {name}")
    except:
        print(f"Error image {name}")
    
