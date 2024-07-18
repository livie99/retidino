import os
from tqdm import tqdm
import pandas as pd 


def split_files(dir):
    ## make train and val directories
    os.mkdir(os.path.join(dir, "train"))
    os.mkdir(os.path.join(dir, "val"))

    ## get all labels / names of folders
    labels = os.listdir(dir)
    labels.sort()
    
    ## access each label
    for label in tqdm(labels):
        if ".csv" in label:
            continue
        
        else:     
            files = os.listdir(os.path.join(dir, label))
            files.sort()
            
            ## split files into train and val
            train = files[:int(len(files)*0.8)]
            val = files[int(len(files)*0.8):]
            
            ## move files to train and val directories
            os.mkdir(os.path.join(dir, "train", label))
            os.mkdir(os.path.join(dir, "val", label))

            for file in tqdm(train):
                os.rename(os.path.join(dir, label, file), os.path.join(dir, "train", label, file))
                
            for file in tqdm(val):
                os.rename(os.path.join(dir, label, file), os.path.join(dir, "val", label, file))
                
            os.rmdir(os.path.join(dir, label))

def split_labels(dir):
    ## define train and val csv files
    train_pd = pd.DataFrame(columns=["label", "name", "n_fields"])
    val_pd = pd.DataFrame(columns=["label", "name", "n_fields"])

    ## read the csv file
    df = pd.read_csv(dir)
    labels = df["label"].unique()

    ## split each label into train and val
    for label in labels:
        df_label = df[(df["label"] == label) & (df["n_fields"] >= 3)]
        
        train = df_label[:int(len(df_label)*0.8)]
        val = df_label[int(len(df_label)*0.8):]

        ## save the train and val split to the csv files
        train_pd = pd.concat([train_pd, train], ignore_index=True)
        val_pd = pd.concat([val_pd, val], ignore_index=True)

    ## save the csv files
    train_pd.to_csv("train_s0.csv", index=False)
    val_pd.to_csv("val_s0.csv", index=False)

    
def main(mode):
    if mode == "data":
        ## need to have access to edit the really data
        dir = "/home/livieymli/retidino/aptos"
        split_files(dir)

    elif mode == "label":
        ## only need access to the label csv file where the labels and the names of the data are stored
        dir = "/home/livieymli/retidino/mosaic/img_nfields.csv"
        split_labels(dir)
    else:
        print("No mode specified or invalid mode")
    pass

if __name__ == "__main__":
    main("label")