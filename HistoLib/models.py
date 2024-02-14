import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from huggingface_hub import from_pretrained_keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2

class RandomBWTransform(tf.keras.layers.Layer):
    def __init__(self, probability=0.5, **kwargs):
        super(RandomBWTransform, self).__init__(**kwargs)
        self.probability = probability

    def call(self, inputs, training=None):
        if training:
            random_number = tf.random.uniform([], minval=0.0, maxval=1.0)

            # Check if the entire batch should be transformed to black and white
            should_transform_bw = random_number < self.probability

            # Convert to grayscale if should_transform_bw is True, else keep the original images
            output = tf.cond(should_transform_bw,
                             lambda: tf.tile(tf.reduce_mean(inputs, axis=-1, keepdims=True), [1, 1, 1, 3]),
                             lambda: inputs)
        else:
            # FIXMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
            output = tf.tile(tf.reduce_mean(inputs, axis=-1, keepdims=True), [1, 1, 1, 3])

        return output

    def get_config(self):
        config = super(RandomBWTransform, self).get_config()
        config.update({'probability': self.probability})
        return config

def augmentation_model():
    # Introduce a layer of data augmentation
    data_augmentation = Sequential([
        # preprocessing.RandomRotation(0.2),
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomFlip("vertical"),
        # preprocessing.RandomZoom(0.2),
        # preprocessing.RandomTranslation(0.2, 0.2),
        # preprocessing.RandomHeight(0.2),
        # preprocessing.RandomWidth(0.2),    
        # preprocessing.RandomContrast(0.2),
        # RandomColorDistortion()
        RandomBWTransform(probability=1),
    ]) 
    return data_augmentation


def simple_model(num_classes, input_shape):
    inp = layers.Input(input_shape)

    data_augmentation = augmentation_model()
    # data_augmentation = Sequential([])
    
    aug = data_augmentation(inp)
    
    # Convolutional
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(aug)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='last_conv')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)

    # Output layer
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    return model, 'SimpleModel'

def simple_model2(num_classes, input_shape):
    inp = layers.Input(input_shape)

    data_augmentation = augmentation_model()
    # data_augmentation = Sequential([])
    
    aug = data_augmentation(inp)
    
    # Convolutional
    x = layers.Conv2D(128, (3, 3), activation='relu')(aug)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', name='last_conv')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)

    # Output layer
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    return model, 'SimpleModel2'


def vit_model(num_classes, input_shape):
    inp = layers.Input(input_shape)
    # res = layers.Resizing(512, 512)(inp)
    
    m1 = from_pretrained_keras("ErnestBeckham/ViT-Lungs")
    
    embedding_layer = m1.get_layer(m1.layers[-2].name)
    m2 = Model(inputs=m1.inputs, outputs=embedding_layer.get_output_at(0))
    
    data_augmentation = augmentation_model()
    # data_augmentation = Sequential([])
    aug = data_augmentation(inp)
    
    x = m2(aug)    
    out = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inp, outputs=out)
    return model, 'ViT'


def efficientnet_model(num_classes, input_shape):
    base_model = EfficientNetB0(include_top = False, weights='imagenet', pooling='avg')

    #capa de entradas. 
    entradas = layers.Input(input_shape)

    # Capa de augmentation
    data_augmentation = augmentation_model()
    x = data_augmentation(entradas)
    # Pass the augmented images through the base model
    x = base_model(x)
    # Add a dense layer
    x = layers.Dense(256, activation='relu')(x)
    # Add another dense layer
    salidas = layers.Dense(num_classes, activation='softmax')(x)
    model1 = Model(inputs = entradas, outputs = salidas)
    
    return model1, 'EfficientNetB0'


def resnet_model(num_classes, input_shape):
    base_model = ResNet50V2(include_top = False, weights='imagenet', pooling='avg')

    #capa de entradas. 
    entradas = layers.Input(input_shape)

    # Capa de augmentation
    data_augmentation = augmentation_model()
    x = data_augmentation(entradas)
    # Pass the augmented images through the base model
    x = base_model(x)
    # Add a dense layer
    x = layers.Dense(256, activation='relu')(x)
    # Add another dense layer
    salidas = layers.Dense(num_classes, activation='softmax')(x)
    model1 = Model(inputs = entradas, outputs = salidas)
    
    return model1, 'ResNet50V2'
        

def get_model(generator, model_name='ResNet50'):
    assert model_name in ['VIT', 'Simple', 'Simple2', 'Eff0', 'ResNet50']
    
    num_classes = generator.num_classes
    input_shape = (*generator.image_size, 3)
    if model_name == 'VIT':
        return vit_model(num_classes, input_shape)
    if model_name == 'Simple':
        return simple_model(num_classes, input_shape)
    if model_name == 'Simple2':
        return simple_model2(num_classes, input_shape)
    if model_name == 'Eff0':
        return efficientnet_model(num_classes, input_shape)
    if model_name == 'ResNet50':
        return resnet_model(num_classes, input_shape)