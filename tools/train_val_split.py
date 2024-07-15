import os
from tqdm import tqdm

dir = "/home/livieymli/retidino/aptos/train_images"

labels = os.listdir(dir)
labels.sort()

os.mkdir(os.path.join(dir, "train"))
os.mkdir(os.path.join(dir, "val"))


for label in tqdm(labels):
    print(label)
    
    files = os.listdir(os.path.join(dir, label))
    files.sort()
    print(len(files))
    
    train = files[:int(len(files)*0.8)]
    val = files[int(len(files)*0.8):]
    print(len(train), len(val))
    
    os.mkdir(os.path.join(dir, "train", label))
    os.mkdir(os.path.join(dir, "val", label))
    for file in tqdm(train):
        os.rename(os.path.join(dir, label, file), os.path.join(dir, "train", label, file))
    for file in tqdm(val):
        os.rename(os.path.join(dir, label, file), os.path.join(dir, "val", label, file))
    os.rmdir(os.path.join(dir, label))
