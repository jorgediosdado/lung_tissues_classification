import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from huggingface_hub import from_pretrained_keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2

def simple_model(num_classes, input_shape):
    inp = layers.Input(input_shape)

    # Convolutional
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(inp)
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

    # Convolutional
    x = layers.Conv2D(128, (3, 3), activation='relu')(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(inp)
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
    
    x = m2(inp)    
    out = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inp, outputs=out)
    return model, 'ViT'


def efficientnet_model(num_classes, input_shape):
    base_model = EfficientNetB0(include_top = False, weights='imagenet', pooling='avg')

    #capa de entradas. 
    entradas = layers.Input(input_shape)

    x = base_model(entradas)
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

    x = base_model(entradas)
    # Add a dense layer
    x = layers.Dense(256, activation='relu', name='prev_dense')(x)
    # Add another dense layer
    x = layers.Dropout(0.5, name='dpout')(x)
    salidas = layers.Dense(num_classes, activation='softmax', name='last_dense')(x)
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