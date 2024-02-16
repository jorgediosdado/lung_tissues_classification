from .utils import get_files, get_dataframe, get_classes_labels, train_test_split

import random
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence, to_categorical


import cv2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightnessContrast, RandomGamma,
    ToFloat, ShiftScaleRotate, ToGray, Normalize, Resize,
    RandomCrop
)

sc = 0.6
sc2 = 0.25
AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    RandomCrop(int(1200*sc), int(sc*1600)),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
                       val_shift_limit=10, p=.9),
    # CLAHE(p=1.0, clip_limit=2.0),
    ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1, 
        rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8), 
    ToGray(p=1),
    # Normalize(),
    Resize(int(1200*sc2), int(sc2*1600)),
    ToFloat(max_value=255),
])

AUGMENTATIONS_TEST = Compose([
    # CLAHE(p=1.0, clip_limit=2.0),
    # Normalize(),
    RandomCrop(int(1200*sc), int(sc*1600)),
    ToGray(p=1),
    Resize(int(1200*sc2), int(sc2*1600)),
    ToFloat(max_value=255),
])

class CustomDataGenerator(Sequence):
    def __init__(self, images, labels, num_classes, augmentations,
                 batch_size=8, 
                 orig_image_size=(1200,1600), 
                 scaled = 0.5,
                 shuffle_epoch=True):
        
        self.num_classes = num_classes
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = tuple([int(s*scaled) for s in orig_image_size])
        self.shuffle_epoch = shuffle_epoch
        self.augment = augmentations
        
    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def __getitem__(self, idx):
        random.seed(17)
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
        # images = images/255
                
        images = np.stack([self.augment(image=x)["image"] for x in images], axis=0)
        images = np.array([self.preprocess_image(im) for im in images])
        
        labels = to_categorical(labels, num_classes=self.num_classes)

        return images, labels
    
    def preprocess_image(self, image):
        # image = tf.image.resize(image, self.image_size).numpy()
        #image = tf.image.random_crop(image, (*self.image_size, 3)).numpy()
        return image
    
    def show_generator(self, N=6):        
        g0 = self[0]
        N = min(N, len(g0[0]))
        fig, axs = plt.subplots(1,N, figsize=(20,4))
        for i in range(N):
            axs[i].imshow(g0[0][i])
            axs[i].axis('off')
            axs[i].set_title(f'Class: {np.argmax(g0[1][i])}')


def get_patient_generators(resolution, 
                           class_type, 
                           exclude_pd, 
                           batch_size=8,
                           root_directory='data/dataset_w_HC/',
                           dataset_csv = 'data/dataset_HC.csv',
                           train_split = 0.6,
                           val_split = 0.2
                          ):
    # Get all the images, filtering by resolution
    image_paths = get_files(root_directory, resolution=resolution, exclude_pd=exclude_pd)
    
    # Get the dataframe of the filtered images
    df = get_dataframe(dataset_csv, image_paths)
    
    # Get the labels associated with each image
    class_names, labels = get_classes_labels(root_directory, df['image_path'].values, class_type, exclude_pd=exclude_pd)
    df['targetclass'] = labels
    
    # Create stratified splits without overlaping pacientes
    df_train, df_test = train_test_split(df, test_size = 1-train_split)
    df_test, df_val = train_test_split(df_test, test_size = round((val_split)/(1-train_split), 3))
    
    # Labels
    train_labels, val_labels, test_labels = df_train['targetclass'].values, df_val['targetclass'].values, df_test['targetclass'].values
    
    # Generators
    train_generator = CustomDataGenerator(df_train['image_path'].values, train_labels, augmentations=AUGMENTATIONS_TRAIN, num_classes=len(class_names), batch_size=batch_size)
    val_generator = CustomDataGenerator(df_val['image_path'].values, val_labels, augmentations=AUGMENTATIONS_TEST, num_classes=len(class_names), shuffle_epoch=False, batch_size=batch_size)
    test_generator = CustomDataGenerator(df_test['image_path'].values, test_labels, augmentations=AUGMENTATIONS_TEST, num_classes=len(class_names), shuffle_epoch=False, batch_size=batch_size)
    
    ##### Debug
    imw, imh = train_generator.image_size
    print(f"{f'Images ({imw}x{imh})':<20}  Training: {len(train_labels):<3} | Validation: {len(val_labels):<3} | Test: {len(test_labels):<3} | Total: {len(labels):<3}")
    print(f"{'Patients':<20}  Training: {len(set(df_train['hc'])):<3} | Validation: {len(set(df_val['hc'])):<3} | Test: {len(set(df_test['hc'])):<3} | Total: {len(set(df['hc'])):<3}")
    
    for tclass in set(labels):
        cs = f'Class {class_names[tclass]:<6} (id {tclass})'
        tr, tv, te = len(train_labels[train_labels==tclass]), len(val_labels[val_labels==tclass]), len(test_labels[test_labels==tclass])
        print(f"{cs:<20}  Training: {tr:<3} | Validation: {tv:<3} | Test: {te:<3} | Total: {tr+tv+te:<3}")
    #####
    
    return train_generator, val_generator, test_generator, class_names