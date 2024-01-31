import os
import random
import imageio
import glob
import datetime
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from IPython.display import display_markdown

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, CenterCrop
from tensorflow.keras.utils import Sequence, to_categorical, plot_model
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tqdm.keras import TqdmCallback
import tensorflow_addons as tfa

from keras import layers
from keras import models
from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score

import io
import cv2
def resize_img_aspect(image, new_height=512):
    height, width, _ = image.shape
    ratio = width / height

    new_width = int(ratio * new_height)

    image = cv2.resize(image, (new_width, new_height))
    return image
    
def fig_to_img(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=3).numpy()    
    return image

# Function to get paths from private directory
def get_files(base_dir, resolution=None, exclude_pd=False):
    resolution = resolution if resolution != 'public' else None
    ext = ['jpg', 'jpeg']
    ret_files = []

    for f in glob.glob(f'{base_dir}/**', recursive=True):
        if not any([f.endswith(e) for e in ext]):
            continue
        if (resolution is not None) and (f'_{resolution}' not in f):
            continue
        if (exclude_pd) and ('_pd' in f):
            continue
        ret_files.append(f)
        
    return ret_files

def get_classes_labels(root_directory, image_paths, class_type, exclude_pd=False):
    if class_type == 'micro':
        class_names = sorted([f for f in os.listdir(root_directory) if not f.startswith('.')])
    else:
        class_names = sorted(list(set([f if '_' not in f else f.split('_')[0] for f in os.listdir(root_directory) if not f.startswith('.')])))
        
    class_names = class_names if not exclude_pd else [c for c in class_names if '_pd' not in c]

    class2int = dict(zip(class_names, range(len(class_names))))
    labels = list(map(lambda im: class2int[im.split(root_directory)[1].split('/')[0]] if class_type=='micro' else class2int[im.split(root_directory)[1].split('/')[0].split('_')[0]], image_paths))
    
    return class_names, class2int, labels
    

class CustomDataGenerator(Sequence):
    def __init__(self, images, labels, num_classes, batch_size=8, image_size=256, 
                 shuffle_epoch=True, mode='train'):
        
        assert mode in ['train', 'val']
        assert batch_size%4 == 0
        
        self.num_classes = num_classes
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle_epoch = shuffle_epoch
        self.mode = mode        
        
    def __len__(self):
        if self.mode == 'train':
            return int(np.ceil(len(self.images) / self.batch_size))
        return int(np.ceil(len(self.images)*4 / self.batch_size))
    
    def __getitem__(self, idx):
        
        if (idx == 0) and (self.shuffle_epoch):            
            # Shuffle at first batch
            c = list(zip(self.images, self.labels))
            random.shuffle(c)
            self.images, self.labels = zip(*c)
            self.images, self.labels = np.array(self.images), np.array(self.labels)       
            
        bs = self.batch_size if self.mode == 'train' else self.batch_size//4
        images = self.images[idx * bs : (idx+1) * bs]
        labels = self.labels[idx * bs : (idx+1) * bs]
        
        # Read images
        images = np.array([imageio.v2.imread(im) for im in images])
        images = images/255
        
        if self.mode == 'train':
            # Choose one of the four quadrants
            x, y = np.random.choice([0,1], size=2)
            #images = images[:,(x*600):(x*600 + 600), (y*800):(y*800 + 800)]

            images = np.array([self.random_crop(im) for im in images])
            labels = to_categorical(labels, num_classes=self.num_classes)

            return images, labels
        
        # new_images, new_labels = [], []
        # for x in range(2):
        #     for y in range(2):
        #         new_images.append(images[:,(x*600):(x*600 + 600), (y*800):(y*800 + 800)])
        #         new_labels.append(to_categorical(labels, num_classes=self.num_classes))

        new_images, new_labels = images, to_categorical(labels, num_classes=self.num_classes)
        # indexes = [i for j in range(4) for i in range(j, len(images) * 4, 4)]
        # indexes = list(range(0,len(images)*4,2)) + list(range(1,len(images)*4,2))
        # print(indexes)
        # new_images = np.concatenate(new_images)[indexes]
        # new_labels = np.concatenate(new_labels)[indexes]
        
        # from ZGlobalLib.visualization import plot_frames
        # plot_frames(images)
        
        new_images = tf.image.resize(new_images, (360, 540)).numpy()
        #new_images = CenterCrop(self.image_size, self.image_size)(new_images).numpy()
        return new_images, new_labels
            
    
    def random_crop(self, image):
        image = tf.image.resize(image, (360, 540)).numpy()
        #cropped_image = tf.image.random_crop(image, size=[self.image_size, self.image_size, 3]).numpy()
        return image
    
    
    def show_generator(self, N=12):        
        g0 = self[0]
        N = min(N, len(g0[0]))
        fig, axs = plt.subplots(1,N, figsize=(20,4))
        for i in range(N):
            axs[i].imshow(g0[0][i])
            axs[i].axis('off')
            axs[i].set_title(g0[1][i])
            

class PublicDataGenerator(Sequence):
    def __init__(self, images, labels, num_classes, batch_size=8, image_size=256, 
                 shuffle_epoch=True, mode='train'):
                
        self.num_classes = num_classes
        self.images = images
        self.labels = labels
        self.batch_size = batch_size*4
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
            
        bs = self.batch_size
        images = self.images[idx * bs : (idx+1) * bs]
        labels = self.labels[idx * bs : (idx+1) * bs]
        
        # Read images
        images = np.array([imageio.v2.imread(im) for im in images])
        images = images/255
        
        images = np.array([self.random_crop(im) for im in images])
        labels = to_categorical(labels, num_classes=self.num_classes)

        return images, labels
            
    
    def random_crop(self, image):
        # cropped_image = tf.image.random_crop(image, size=[self.image_size, self.image_size, 3]).numpy()
        cropped_image = tf.image.resize(image, (256, 256)).numpy()
        return cropped_image
    
    
    def show_generator(self, N=12):        
        g0 = self[0]
        N = min(N, len(g0[0]))
        fig, axs = plt.subplots(1,N, figsize=(20,4))
        for i in range(N):
            axs[i].imshow(g0[0][i])
            axs[i].axis('off')
            axs[i].set_title(g0[1][i])
    
    
def get_generators(image_paths, labels, num_classes, resolution, test_size=0.15, batch_size=8, random_state=42):
    #Split in training and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=test_size, random_state=42)
 
    #Build the generators
    if resolution == 'public':
        train_generator = PublicDataGenerator(train_paths, train_labels, num_classes=num_classes, batch_size=batch_size)
        val_generator = PublicDataGenerator(val_paths, val_labels, num_classes=num_classes, shuffle_epoch=False, batch_size=batch_size)
    else:        
        train_generator = CustomDataGenerator(train_paths, train_labels, num_classes=num_classes, batch_size=batch_size)
        val_generator = CustomDataGenerator(val_paths, val_labels, num_classes=num_classes, shuffle_epoch=False, mode='val', batch_size=batch_size)
    
    return train_generator, val_generator


from sklearn.model_selection import GroupShuffleSplit
# from sklearn.model_selection import StratifiedShuffleSplit

# class StratifiedGroupShuffleSplit:
#     def __init__(self, n_splits=10, test_size=0.2, random_state=None):
#         self.n_splits = n_splits
#         self.test_size = test_size
#         self.random_state = random_state

#     def split(self, X, y, groups):
#         sss = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=self.test_size, random_state=self.random_state)

#         for train_index, test_index in sss.split(X, y):
#             train_groups = groups.iloc[train_index]
#             test_groups = groups.iloc[test_index]
            
#             yield train_index, test_index

def cust_split(df, test_size=0.5):
    groups = df.groupby('targetclass')
    all_train = []
    all_test = []
    for group_id, group in groups:
        # if a group is already taken in test or train it must stay there
        group = group[~group['hc'].isin(all_train+all_test)]
        # if group is empty 
        if group.shape[0] == 0:
            continue
        train_inds, test_inds = next(GroupShuffleSplit(
            test_size=test_size, n_splits=2, random_state=7).split(group, groups=group['hc']))

        all_train += group.iloc[train_inds]['hc'].tolist()
        all_test += group.iloc[test_inds]['hc'].tolist()

    df_train= df[df['hc'].isin(all_train)]
    df_test= df[df['hc'].isin(all_test)]
    
    return df_train, df_test

def get_patient_generators(resolution, class_type, exclude_pd, 
                           batch_size=8,
                           root_directory='data/dataset_w_HC/',
                          ):
    image_paths = get_files(root_directory, resolution=resolution, exclude_pd=exclude_pd)
    df = pd.read_csv('data/dataset_HC.csv')
    df['image_path'] = ""
    used = set()
    for idx, row in df.iterrows():
        for idim, im in enumerate(image_paths):
            if (row['superclass'] in im) and ((pd.isnull(row['subclass'])) or (row['subclass'] in im)) and (row['resolution'] in im) and ('_'+row['image_id']+'.jpg' in im):
                df.loc[idx, 'image_path'] = im    
                if idim in used:                
                    print(idx, row, idim, im)
                    assert False
                used.add(idim)
                break
    df = df[df['hc']!=0].reset_index(drop=True)
    df = df[df['image_path']!=""].reset_index(drop=True)
    df = df.sample(frac=1)
    
    image_paths = df['image_path'].values
    
    class_names, class2int, labels = get_classes_labels(root_directory, image_paths, class_type, exclude_pd=exclude_pd)
    num_classes = len(class2int)
    df['targetclass'] = labels
    
#     sp = 0.5
#     train_index, test_index = [], []
#     for lab in class2int.values():
#         dftemp = df[df['targetclass']==lab]
#         group_splitter = GroupShuffleSplit(n_splits=1, test_size=sp, random_state=42)
#         tr, te = next(group_splitter.split(dftemp, groups=dftemp['hc']))
#         train_index.append(dftemp.iloc[tr].index.values)
#         test_index.append(dftemp.iloc[te].index.values)
#     train_index = np.concatenate(train_index)
#     test_index = np.concatenate(test_index)
    
#     df_train, df_test = df.loc[train_index], df.loc[test_index]
#     train_labels = df_train['targetclass'].values
    
#     sp = 0.5
#     val_index, test_index = [], []
#     for lab in class2int.values():
#         dftemp = df_test[df_test['targetclass']==lab]
#         group_splitter = GroupShuffleSplit(n_splits=1, test_size=sp, random_state=42)
#         tv, te = next(group_splitter.split(dftemp, groups=dftemp['hc']))
#         val_index.append(dftemp.iloc[tv].index.values)
#         test_index.append(dftemp.iloc[te].index.values)
#     val_index = np.concatenate(val_index)
#     test_index = np.concatenate(test_index)
    
#     df_val, df_test = df_test.loc[val_index], df_test.loc[test_index]
#     val_labels = df_val['targetclass'].values
#     test_labels = df_test['targetclass'].values
    
    
    df_train, df_test = cust_split(df, test_size=0.3)
    df_val, df_test = cust_split(df_test, test_size=0.5)
    
    train_labels, val_labels, test_labels = df_train['targetclass'].values, df_val['targetclass'].values, df_test['targetclass'].values
    
#     group_splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

#     for train_index, test_index in group_splitter.split(image_paths, labels, groups=df['hc']):
#         df_train, df_test = df.iloc[train_index], df.iloc[test_index]
#         train_labels, test_labels = np.array(labels)[train_index], np.array(labels)[test_index]

#     group_splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
#     for val_index, test_index in group_splitter.split(df_test['image_path'], test_labels, groups=df_test['hc']):
#         df_val, df_test = df.iloc[val_index], df.iloc[test_index]
#         val_labels, test_labels = test_labels[val_index], test_labels[test_index]

    
    train_generator = CustomDataGenerator(df_train['image_path'], train_labels, num_classes=num_classes, batch_size=batch_size)
    val_generator = CustomDataGenerator(df_val['image_path'], val_labels, num_classes=num_classes, shuffle_epoch=False, batch_size=batch_size)
    test_generator = CustomDataGenerator(df_test['image_path'], test_labels, num_classes=num_classes, shuffle_epoch=False, batch_size=batch_size)
    
    return train_generator, val_generator, test_generator
    #return df_train, train_labels, df_val, val_labels, df_test, test_labels
        

def compute_weights(train_generator):
    labels = np.concatenate([l.argmax(1) for _, l in tqdm(train_generator, leave=False)])
    class_weights = class_weight.compute_class_weight('balanced',
                                                         classes=np.unique(labels),
                                                         y=list(labels))
    class_weights = dict(enumerate(class_weights))
    return class_weights



def simple_model(num_classes, resolution):
    entradas = layers.Input((360, 540, 3))

    data_augmentation = Sequential([
        preprocessing.RandomRotation(0.2),
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomFlip("vertical"),
        preprocessing.RandomZoom(0.2),
        preprocessing.RandomTranslation(0.2, 0.2),
        preprocessing.RandomHeight(0.2),
        preprocessing.RandomWidth(0.2),    
        preprocessing.RandomContrast(0.2),
    ])
    # data_augmentation = Sequential([])
    aug = data_augmentation(entradas)
    
    # Two convolutional layers with 16 filters each
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(aug)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    x = layers.GlobalAveragePooling2D()(x)

    # Dense layer with 1280 units
    x = layers.Dense(256, activation='relu')(x)

    # Output layer
    output_tensor = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=entradas, outputs=output_tensor)
    return model

    
def get_model(num_classes, resolution):
    
    return simple_model(num_classes, resolution)
    
    base_model = EfficientNetB0(include_top = False, weights='imagenet', pooling='avg')
    # base_model = EfficientNetB0(include_top = False, weights=None, pooling='avg')

    # Introduce a layer of data augmentation
    data_augmentation = Sequential([
        preprocessing.RandomRotation(0.2),
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomFlip("vertical"),
        preprocessing.RandomZoom(0.2),
        preprocessing.RandomTranslation(0.2, 0.2),
        preprocessing.RandomHeight(0.2),
        preprocessing.RandomWidth(0.2),    
        preprocessing.RandomContrast(0.2),

    ]) if resolution != 'public' else Sequential([])

    # data_augmentation = Sequential([])
    
    # # Freeze all layers in the base model
    # for layer in base_model.layers:
    #     layer.trainable = False
    # # Unfreeze the last 10 layers in the base model for fine-tuning
    # for layer in base_model.layers[-5:]:
    #     layer.trainable = True

    #capa de entradas. 
    entradas = layers.Input((360, 540, 3))

    # Capa de augmentation
    x = data_augmentation(entradas)
    # Pass the augmented images through the base model
    x = base_model(x)
    # Add a dense layer
    x = layers.Dense(512, activation='relu')(x)
    # Add another dense layer
    salidas = layers.Dense(num_classes, activation='softmax')(x)
    model1 = Model(inputs = entradas, outputs = salidas)
    
    return model1

from ZGlobalLib.callbacks import KeyboardInterruptCallback, SimpleLogger
def train_model(model, train_generator, val_generator, num_classes, class_weights, log_dir):
    num_epochs = 2000
    patience = 100
    patience_lr = 50
    
    init_lr = 1e-4

    model.compile(optimizer=tf.keras.optimizers.Adam(init_lr), 
                  loss='categorical_crossentropy', 
                  metrics=[
                        tf.keras.metrics.CategoricalAccuracy(name=f'metrics/accuracy'),
                        tf.keras.metrics.TopKCategoricalAccuracy(3, name=f'metrics/top-3-accuracy'),
                        tfa.metrics.F1Score(num_classes=num_classes, average='macro', name='metrics/F1-macro'),
                        tf.keras.metrics.AUC(multi_label=True, num_labels=num_classes, name='metrics/AUC'),
                        tf.keras.metrics.Precision(name='metrics/precision'),
                        tf.keras.metrics.Recall(name='metrics/recall'),
                        tf.keras.metrics.PrecisionAtRecall(0.99, name='metrics/P@R_99'),
                        tf.keras.metrics.PrecisionAtRecall(0.95, name='metrics/P@R_95'),
                        tf.keras.metrics.PrecisionAtRecall(0.9, name='metrics/P@R_90'),
                        tfa.metrics.MatthewsCorrelationCoefficient(num_classes=num_classes, name='metrics/MCC')
                    ],
                 )

    callbacks =[
           SimpleLogger(log_dir, show_model=False),
           EarlyStopping(monitor='val_loss', restore_best_weights=False, patience=patience),
           ReduceLROnPlateau(monitor='val_loss', patience=patience_lr, min_lr=1e-7),       
           ModelCheckpoint(log_dir, monitor=f"val_loss", save_best_only=True, save_weights_only=True),
           TqdmCallback(leave=False),
           KeyboardInterruptCallback()
    ]
    
    history = model.fit(train_generator, epochs=num_epochs, verbose=0, callbacks=callbacks, validation_data=val_generator,class_weight=class_weights)
    
    model.load_weights(log_dir)
    
    return history

#function to plot the metrics of the trainign and validation
def plot_metrics(history, log_dir):
    # Plotting training accuracy
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    axs[0].plot(history.history['metrics/accuracy'], label='Train')
    axs[0].plot(history.history['val_metrics/accuracy'], label='Validation')
    axs[0].set_title('Training and Validation Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # Plotting training loss
    axs[1].plot(history.history['loss'], label='Train')
    axs[1].plot(history.history['val_loss'], label='Validation')
    axs[1].set_title('Training and Validation Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.tight_layout()
    
    image = fig_to_img(fig)
    image = resize_img_aspect(image, 512)
    plt.imsave(os.path.join(log_dir, 'plot.png'), image)
    
    plt.show()
    
    
    
def get_test_generator(class2int, resolution, log_dir, test_directory = "data/validation_final_septiembre/", exclude_pd=False):
    test_image_paths = get_files(test_directory, resolution=resolution, exclude_pd=exclude_pd)    
    num_classes = len(class2int)
    class_names = sorted(class2int.keys())
    test_labels = list(map(lambda im: class2int[im.split(test_directory)[1].split('/')[0]] if num_classes in [5, 7] else class2int[im.split(test_directory)[1].split('/')[0].split('_')[0]], test_image_paths))

    # Test labels are repeated 4 times since each image is divided in 4 patches
    # test_labels = np.repeat(np.expand_dims(test_labels,0), 4, 0).T.flatten()
    # Extract image paths and labels

    test_generator = CustomDataGenerator(test_image_paths, test_labels, num_classes, shuffle_epoch=False, mode='val', batch_size=8)
    
    return test_generator
    
def test_model(model, test_generator, log_dir, class_names):

    test_predictions = model.predict(test_generator)
    test_labels = np.concatenate([np.argmax(t[1], 1) for t in test_generator])

    # Convert predictions to class labels
    predicted_labels = np.argmax(test_predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predicted_labels)
    precision = precision_score(test_labels, predicted_labels, average='macro')
    recall = recall_score(test_labels, predicted_labels, average='macro')

    print("Test Accuracy:", accuracy)
    print("Test Precision:", precision)
    print("Test Recall:", recall)

    # Obtain the confusion matrix

    # Ensure that labels are unique and match the confusion matrix
    labels = np.unique(np.concatenate((test_labels, predicted_labels)))

    # Visualize the confusion matrix
    fig = plt.figure(figsize=(5,5))
    ax = fig.gca()
    ConfusionMatrixDisplay.from_predictions(test_labels, predicted_labels, display_labels=class_names, normalize='true', ax=ax)
    
    ax.set_title(f'Acc: {accuracy:4.2f}')
    image = fig_to_img(fig)
    image = resize_img_aspect(image, 512)
    plt.imsave(os.path.join(log_dir, 'confusion.png'), image)
    
    plt.show()

# def train_evaluate(class_type, resolution,
#                    public_directory = 'data/public_dataset/',
#                    root_directory = "data/dataset_2_final/", 
#                    test_directory = "data/validation_final_septiembre/",
#                    pretrain_dir = None,
#                    exclude_pd = False
#                   ):
    
#     if resolution == 'public':
#         root_directory = public_directory
    
#     resname = resolution if resolution is not None else 'all'
    
#     image_paths = get_files(root_directory, resolution=resolution, exclude_pd=exclude_pd)
#     class_names, class2int, labels = get_classes_labels(root_directory, image_paths, class_type, exclude_pd=exclude_pd)
    
#     num_classes = len(class2int)
    
#     display_markdown(f'## Evaluating {resname} resolution, {num_classes} classes, exc pd {exclude_pd}', raw=True)
    
#     train_generator, val_generator = get_generators(image_paths, labels, num_classes=num_classes, resolution=resolution)
#     class_weights = compute_weights(train_generator)

#     model = get_model(num_classes, resolution=resolution)
    
#     if pretrain_dir is not None:
#         model.load_weights(pretrain_dir)
    
#     MODEL_NAME = f'Ef0_{resname}_{num_classes}_classes_excpd{int(exclude_pd)}'
#     RUN_NAME = ''
#     log_dir = f'logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/{MODEL_NAME}{RUN_NAME}'
#     print(log_dir)
    
#     history = train_model(model, train_generator, val_generator, num_classes, class_weights, log_dir)
    
#     plot_metrics(history, log_dir)
#     test_model(model, class2int, resolution, log_dir, exclude_pd=exclude_pd)