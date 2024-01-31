from .utils import get_files, get_dataframe, get_classes_labels, train_test_split

import random
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence, to_categorical

class CustomDataGenerator(Sequence):
    def __init__(self, images, labels, num_classes, 
                 batch_size=8, 
                 image_size=(360,540), 
                 shuffle_epoch=True):
        
        self.num_classes = num_classes
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle_epoch = shuffle_epoch
        
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
        images = images/255
        
        images = np.array([self.preprocess_image(im) for im in images])
        labels = to_categorical(labels, num_classes=self.num_classes)

        return images, labels
    
    def preprocess_image(self, image):
        image = tf.image.resize(image, self.image_size).numpy()
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
                           train_split = 0.7,
                           val_split = 0.15
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
    train_generator = CustomDataGenerator(df_train['image_path'].values, train_labels, num_classes=len(class_names), batch_size=batch_size)
    val_generator = CustomDataGenerator(df_val['image_path'].values, val_labels, num_classes=len(class_names), shuffle_epoch=False, batch_size=batch_size)
    test_generator = CustomDataGenerator(df_test['image_path'].values, test_labels, num_classes=len(class_names), shuffle_epoch=False, batch_size=batch_size)
    
    return train_generator, val_generator, test_generator, class_names