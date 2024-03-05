from .utils import get_files, get_dataframe, get_classes_labels, train_test_split

import random
import imageio
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence, to_categorical

# !pip install albumentations
import cv2
import albumentations as A

def train_augmentations(percent_resize=0.25):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GridDistortion(p=0.2),
        A.RandomSizedCrop(min_max_height=(750, 1200), height=1200, width=1600, p=0.4),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
                           val_shift_limit=10, p=.2),
        A.Resize(int(1200*percent_resize), int(percent_resize*1600)),
        A.ToFloat(max_value=255),
    ])

def test_augmentations(percent_resize=0.25):
    return A.Compose([
    A.Resize(int(1200*percent_resize), int(percent_resize*1600)),
    A.ToFloat(max_value=255),
])

class CustomDataGenerator(Sequence):
    def __init__(self, images, labels, num_classes, augmentations,
                 batch_size=8, 
                 shuffle_epoch=True):
        
        self.num_classes = num_classes
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle_epoch = shuffle_epoch
        self.augment = augmentations
        
        random.seed(17)
        
    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def __getitem__(self, idx):
        
        if (idx == 0) and (self.shuffle_epoch):            
            # Shuffle at first batch
            c = list(zip(self.images, self.labels))
            random.shuffle(c)
            self.images, self.labels = zip(*c)
            self.images, self.labels = np.array(self.images), np.array(self.labels)       
            
        # Get one batch
        bs = self.batch_size
        images = self.images[idx * bs : (idx+1) * bs]
        labels = self.labels[idx * bs : (idx+1) * bs]
        
        # Read images
        images = np.array([imageio.v3.imread(im) for im in images])                
        images = np.stack([self.augment(image=x)["image"] for x in images], axis=0)        
        labels = to_categorical(labels, num_classes=self.num_classes)

        return images, labels

    
    def show_generator(self, N=6):  
        used = set()
        fig, axs = plt.subplots(1,N, figsize=(20,4))
        for i in range(N):
            batch_idx = np.random.randint(0, len(self))
            batch = self[batch_idx]
            img_idx = np.random.randint(0, len(batch))
            while (batch_idx, img_idx) in used:
                batch_idx = np.random.randint(0, len(self))
                batch = self[batch_idx]
                img_idx = np.random.randint(0, len(batch))
            used.add((batch_idx, img_idx))
            img = batch[0][img_idx]
            axs[i].imshow(img)
            axs[i].axis('off')
            axs[i].set_title(f'Class: {np.argmax(batch[1][img_idx])}')

def get_patient_generators(resolution, 
                           class_type, 
                           exclude_pd, 
                           batch_size=8,
                           root_directory='data/dataset_w_HC/',
                           dataset_csv = 'data/dataset_HC.csv',
                           train_split = 0.8,
                           val_split = 0.1,
                           random_state = 967,
                           image_scale = 0.25,
                           debug = False
                          ):

    # Get the dataframe of the filtered images
    df = get_dataframe(dataset_csv, resolution=resolution, exclude_pd=exclude_pd)
    df = df[df.hc != 0]
    
    # Get the labels associated with each image
    class_names, labels = get_classes_labels(root_directory, df['image_path'].values, class_type, exclude_pd=exclude_pd)
    df['targetclass'] = labels
    
    # Create stratified splits without overlaping pacientes
    df_train, df_test = train_test_split(df, test_size = 1-train_split, random_state=random_state)
    df_test, df_val = train_test_split(df_test, test_size = round((val_split)/(1-train_split), 3), random_state=random_state)
    
    # Labels
    train_labels, val_labels, test_labels = df_train['targetclass'].values, df_val['targetclass'].values, df_test['targetclass'].values
    
    # Generators
    train_generator = CustomDataGenerator(df_train['image_path'].values, train_labels, augmentations=train_augmentations(percent_resize=image_scale), num_classes=len(class_names), batch_size=batch_size)
    val_generator = CustomDataGenerator(df_val['image_path'].values, val_labels, augmentations=test_augmentations(percent_resize=image_scale), num_classes=len(class_names), shuffle_epoch=False, batch_size=batch_size)
    test_generator = CustomDataGenerator(df_test['image_path'].values, test_labels, augmentations=test_augmentations(percent_resize=image_scale), num_classes=len(class_names), shuffle_epoch=False, batch_size=batch_size)
    
    ##### Debug
    if debug:
        imw, imh, _ = train_generator[0][0][0].shape
        #imw, imh = int(1200*sc2), int(sc2*1600)
        print(f"{f'Images ({imw}x{imh})':<20}  Training: {len(train_labels):<3} | Validation: {len(val_labels):<3} | Test: {len(test_labels):<3} | Total: {len(labels):<3}")
        print(f"{'Patients':<20}  Training: {len(set(df_train['hc'])):<3} | Validation: {len(set(df_val['hc'])):<3} | Test: {len(set(df_test['hc'])):<3} | Total: {len(set(df['hc'])):<3}")

        for tclass in set(labels):
            cs = f'Class {class_names[tclass]:<6} (id {tclass})'
            tr, tv, te = len(train_labels[train_labels==tclass]), len(val_labels[val_labels==tclass]), len(test_labels[test_labels==tclass])
            print(f"{cs:<20}  Training: {tr:<3} | Validation: {tv:<3} | Test: {te:<3} | Total: {tr+tv+te:<3}")
    #####
    
    return train_generator, val_generator, test_generator, class_names