from .callbacks import KeyboardInterruptCallback, SimpleLogger

import os
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

def get_logdir(model_name, run_name='', base_log='logs'):
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = model_name+run_name
    
    return os.path.join(base_log, date, run_name)


def compile_model(model, num_classes, init_lr=1e-4):
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
    return model


def train_model(model, train_generator, val_generator, class_weights, log_dir, 
                num_epochs = 2000, 
                patience = 100,
                patience_lr = 50):

    callbacks =[
           SimpleLogger(log_dir, show_model=False),
           EarlyStopping(monitor='val_loss', restore_best_weights=False, patience=patience),
           ReduceLROnPlateau(monitor='val_loss', patience=patience_lr, min_lr=1e-7),       
           ModelCheckpoint(log_dir, monitor=f"val_loss", save_best_only=True, save_weights_only=True),
           TqdmCallback(leave=False),
           KeyboardInterruptCallback()
    ]
    
    history = model.fit(train_generator, epochs=num_epochs, verbose=0, callbacks=callbacks, validation_data=val_generator,class_weight=class_weights)
    
    print(f'Loading weights with best iteration...')
    model.load_weights(log_dir)
    
    return history


def plot_metrics(history):
    # Plotting training accuracy
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    axs[0].plot(history.history['metrics/AUC'], label='Train')
    axs[0].plot(history.history['val_metrics/AUC'], label='Validation')
    axs[0].set_title('Training and Validation AUC')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('AUC')
    axs[0].legend()

    # Plotting training loss
    axs[1].plot(history.history['loss'], label='Train')
    axs[1].plot(history.history['val_loss'], label='Validation')
    axs[1].set_title('Training and Validation Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    
    
def test_model(model, test_generator, class_names):

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

    # Ensure that labels are unique and match the confusion matrix
    labels = np.unique(np.concatenate((test_labels, predicted_labels)))

    # Visualize the confusion matrix
    fig = plt.figure(figsize=(5,5))
    ax = fig.gca()
    ConfusionMatrixDisplay.from_predictions(test_labels, predicted_labels, display_labels=class_names, normalize='true', ax=ax)
    
    ax.set_title(f'Acc: {accuracy:4.2f}')
    plt.show()