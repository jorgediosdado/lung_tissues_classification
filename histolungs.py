import os
import os
import random
import imageio
import glob
import datetime
import numpy as np
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

# Function to get paths from private directory
def get_files(base_dir, resolution=None):
    ext = ['jpg', 'jpeg']
    ret_files = []

    for f in glob.glob(f'{base_dir}/**', recursive=True):
        if not any([f.endswith(e) for e in ext]):
            continue
        if (resolution is not None) and (f'_{resolution}' not in f):
            continue
        ret_files.append(f)
        
    return ret_files

def get_classes_labels(root_directory, image_paths, num_classes):
    if num_classes == 7:
        class_names = sorted([f for f in os.listdir(root_directory) if not f.startswith('.')])
    else:
        class_names = sorted(list(set([f if '_' not in f else f.split('_')[0] for f in os.listdir(root_directory) if not f.startswith('.')])))

    class2int = dict(zip(class_names, range(len(class_names))))
    labels = list(map(lambda im: class2int[im.split(root_directory)[1].split('/')[0]] if num_classes==7 else class2int[im.split(root_directory)[1].split('/')[0].split('_')[0]], image_paths))
    
    return class_names, class2int, labels
    

class CustomDataGenerator(Sequence):
    def __init__(self, images, labels, num_classes, batch_size=8, image_size=255, 
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
        
        if self.mode == 'train':
            # Choose one of the four quadrants
            x, y = np.random.choice([0,1], size=2)
            images = images[:,(x*600):(x*600 + 600), (y*800):(y*800 + 800)]

            #Todos los modelos tienen capa de rescaling por lo que no es necesaria aqui
            # images = images/255

            images = np.array([self.random_crop(im) for im in images])
            labels = to_categorical(labels, num_classes=self.num_classes)

            return images, labels
        
        new_images, new_labels = [], []
        for x in range(2):
            for y in range(2):
                new_images.append(images[:,(x*600):(x*600 + 600), (y*800):(y*800 + 800)])
                new_labels.append(to_categorical(labels, num_classes=self.num_classes))

        s = len(new_images[0])
        # indexes = list(range(0,len(images)*4,4))+list(range(1,len(images)*4,4))+list(range(2,len(images)*4,4))+list(range(3,len(images)*4,4))
        indexes = [i for j in range(4) for i in range(j, len(images) * 4, 4)]
        new_images = np.concatenate(new_images)[indexes]
        new_labels = np.concatenate(new_labels)[indexes]
                
        new_images = CenterCrop(self.image_size, self.image_size)(new_images).numpy().astype(int)
        return new_images, new_labels
            
    
    def random_crop(self, image):
        cropped_image = tf.image.random_crop(image, size=[self.image_size, self.image_size, 3]).numpy()
        return cropped_image
    
    
    def show_generator(self, N=12):        
        g0 = self[0]
        N = min(N, len(g0[0]))
        fig, axs = plt.subplots(1,N, figsize=(20,4))
        for i in range(N):
            axs[i].imshow(g0[0][i])
            axs[i].axis('off')
            axs[i].set_title(g0[1][i])
            
    
    
def get_generators(image_paths, labels, num_classes, test_size=0.15, batch_size=32, random_state=42):
    #Split in training and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=test_size, random_state=42)

    #Build the generators
    train_generator = CustomDataGenerator(train_paths, train_labels, num_classes=num_classes, batch_size=batch_size)
    val_generator = CustomDataGenerator(val_paths, val_labels, num_classes=num_classes, shuffle_epoch=False, mode='val', batch_size=batch_size)
    
    return train_generator, val_generator


def compute_weights(train_generator):
    labels = np.concatenate([l.argmax(1) for _, l in tqdm(train_generator, leave=False)])
    class_weights = class_weight.compute_class_weight('balanced',
                                                         classes=np.unique(labels),
                                                         y=list(labels))
    class_weights = dict(enumerate(class_weights))
    return class_weights

    
def get_model(num_classes):
    base_model = EfficientNetB0(include_top = False ,weights='imagenet', pooling='avg')

    # Introduce a layer of data augmentation
    data_augmentation = Sequential([
        preprocessing.RandomRotation(0.2),
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomFlip("vertical"),
        preprocessing.RandomZoom(0.2),
        preprocessing.RandomContrast(0.2),
        preprocessing.RandomTranslation(0.2, 0.2),
        preprocessing.RandomHeight(0.2),
        preprocessing.RandomWidth(0.2),    
        preprocessing.RandomContrast(0.2),

    ])

    # # Freeze all layers in the base model
    # for layer in base_model.layers:
    #     layer.trainable = False
    # # Unfreeze the last 10 layers in the base model for fine-tuning
    # for layer in base_model.layers[-5:]:
    #     layer.trainable = True

    #capa de entradas. 
    entradas = layers.Input((255, 255, 3))

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


def train_model(model, train_generator, val_generator, num_classes, class_weights, log_dir):
    num_epochs = 200
    patience = 40
    patience_lr = 20

    model.compile(optimizer=tf.keras.optimizers.Adam(), 
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
           EarlyStopping(monitor='val_loss', restore_best_weights=False, patience=patience),
           ReduceLROnPlateau(monitor='val_loss', patience=patience_lr, min_lr=1e-7),       
           ModelCheckpoint(log_dir, monitor=f"val_loss", save_best_only=True, save_weights_only=True),
           TqdmCallback(leave=False),
           TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)

    ]
    
    history = model.fit(train_generator, epochs=num_epochs, verbose=0, callbacks=callbacks, validation_data=val_generator,class_weight=class_weights)
    
    model.load_weights(log_dir)
    
    return history

#function to plot the metrics of the trainign and validation
def plot_metrics(history):
    # Plotting training accuracy
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['metrics/accuracy'], label='Train')
    plt.plot(history.history['val_metrics/accuracy'], label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting training loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    
def test_model(model, class2int, resolution, test_directory = "data/validation_final_septiembre/"):

    test_image_paths = get_files(test_directory, resolution=resolution)
    num_classes = len(class2int)
    class_names = sorted(class2int.keys())
    test_labels = list(map(lambda im: class2int[im.split(test_directory)[1].split('/')[0]] if num_classes==7 else class2int[im.split(test_directory)[1].split('/')[0].split('_')[0]], test_image_paths))

    # Test labels are repeated 4 times since each image is divided in 4 patches
    test_labels = np.repeat(np.expand_dims(test_labels,0), 4, 0).T.flatten()
    # Extract image paths and labels

    test_generator = CustomDataGenerator(test_image_paths, test_labels, num_classes, shuffle_epoch=False, mode='val', batch_size=8)
    test_predictions = model.predict(test_generator)

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
    ax = plt.figure(figsize=(5,5)).gca()
    ConfusionMatrixDisplay.from_predictions(test_labels, predicted_labels, display_labels=class_names, normalize='true', ax=ax)
    plt.show()

def train_evaluate(num_classes, resolution, 
                   root_directory = "data/dataset_2_final/", 
                   test_directory = "data/validation_final_septiembre/"):
    
    resname = resolution if resolution is not None else 'all'
    # print(f'Evaluating {resname} resolution, {num_classes} classes')
    display_markdown(f'## Evaluating {resname} resolution, {num_classes} classes', raw=True)
    
    image_paths = get_files(root_directory, resolution=resolution)
    class_names, class2int, labels = get_classes_labels(root_directory, image_paths, num_classes)
    train_generator, val_generator = get_generators(image_paths, labels, num_classes=num_classes)
    class_weights = compute_weights(train_generator)

    model = get_model(num_classes)
    
    MODEL_NAME = f'Ef0_{resname}_{num_classes}_classes'
    RUN_NAME = ''
    log_dir = f'logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/{MODEL_NAME}{RUN_NAME}'
    
    history = train_model(model, train_generator, val_generator, num_classes, class_weights, log_dir)
    
    plot_metrics(history)
    test_model(model, class2int, resolution)