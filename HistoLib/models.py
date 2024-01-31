from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers.experimental import preprocessing

def simple_model(num_classes):
    inp = layers.Input((360, 540, 3))

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


def get_model(num_classes):
    return simple_model(num_classes)
    
#     base_model = EfficientNetB0(include_top = False, weights='imagenet', pooling='avg')
#     # base_model = EfficientNetB0(include_top = False, weights=None, pooling='avg')

#     # Introduce a layer of data augmentation
#     data_augmentation = Sequential([
#         preprocessing.RandomRotation(0.2),
#         preprocessing.RandomFlip("horizontal"),
#         preprocessing.RandomFlip("vertical"),
#         preprocessing.RandomZoom(0.2),
#         preprocessing.RandomTranslation(0.2, 0.2),
#         preprocessing.RandomHeight(0.2),
#         preprocessing.RandomWidth(0.2),    
#         preprocessing.RandomContrast(0.2),

#     ]) if resolution != 'public' else Sequential([])

#     #capa de entradas. 
#     entradas = layers.Input((360, 540, 3))

#     # Capa de augmentation
#     x = data_augmentation(entradas)
#     # Pass the augmented images through the base model
#     x = base_model(x)
#     # Add a dense layer
#     x = layers.Dense(512, activation='relu')(x)
#     # Add another dense layer
#     salidas = layers.Dense(num_classes, activation='softmax')(x)
#     model1 = Model(inputs = entradas, outputs = salidas)
    
#     return model1, 'EfficientNetB0'