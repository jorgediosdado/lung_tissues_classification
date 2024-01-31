import io
import signal
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from matplotlib.image import imread
from tensorflow.keras.utils import plot_model

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


class SimpleLogger(keras.callbacks.Callback):
    def __init__(self, log_dir, show_model=True, expand_nested=True, 
                 optimizer_name='optimizer', lr_name='details/learning-rate', *args, **kwargs):
        """
        It logs some metrics to be displayed at the Tensorboard UI.
        
        Params
        ======
        :log_dir: The path of the log file.
        :show_model: Whether or not to store an image of the current model
        :expand_nested: If show_model is True, this will expand the submodels the model contains.
        :optimizer_name: String (for a single value) or List for a multiple values. It contains the name 
                         of the optimizers we want to log.
        :lr_name: String (for a single value) or List for a multiple values. For each optimizer in the 
                  :optimizer_name: list, a log will be created.        
        """
        super().__init__(*args, **kwargs)
        
        # To save the model architecture as a PNG image
        self.show_model = show_model
        self.expand_nested = expand_nested
        
        # The optimizer that have the learning rate we want to display.
        # TODO: Multiple learning rates
        self.optimizer_name = optimizer_name if type(optimizer_name) == list else [optimizer_name]
        self.lr_name = lr_name if type(lr_name) == list else [lr_name]
        
        # Writters for the metrics and values
        self.train_writer = tf.summary.create_file_writer(log_dir+'/train')
        self.val_writer = tf.summary.create_file_writer(log_dir+'/val')

        assert len(self.optimizer_name) == len(self.lr_name), 'Different amount of optimizers and names'
    
    def on_train_begin(self, logs=None):
        
        if not self.show_model:
            return
        
        mydpi = 200
        plot_model(self.model, to_file='model.png', show_shapes=True, dpi=mydpi, expand_nested=self.expand_nested)
        
        im = imread('model.png')
        h, w, _ = im.shape
        fig = plt.figure(figsize=(w/mydpi, h/mydpi))
        ax = fig.gca()
        ax.imshow(im)
        ax.axis('off')
        plt.tight_layout()
        
        cm_image = plot_to_image(fig)

        # Log the confusion matrix as an image summary.
        with self.train_writer.as_default():
            tf.summary.image("2. Metrics/Model", cm_image, step=0)
            self.train_writer.flush()
        
    
    def on_epoch_begin(self, epoch, logs=None):
        # Learning rate logger        
        for optimizer_name, lr_name in zip(self.optimizer_name, self.lr_name):
            model_optimizer = getattr(self.model, optimizer_name)
            try:
                lr = float(tf.keras.backend.get_value(model_optimizer.learning_rate))
            except:
                lr = float(tf.keras.backend.get_value(model_optimizer.lr(model_optimizer.iterations)))        

            for writer in [self.train_writer, self.val_writer]:
                with writer.as_default():
                    tf.summary.scalar(lr_name, data=lr, step=epoch)        
                    writer.flush()

    def on_epoch_end(self, epoch, logs=None):              
        # Log metrics
        for k,v in logs.items():
            writer = self.val_writer if k.startswith('val_') else self.train_writer
            name = k[4:] if k.startswith('val_') else k
            with writer.as_default():
                tf.summary.scalar(name, data=v, step=epoch)
                writer.flush()
                
class KeyboardInterruptCallback(keras.callbacks.Callback):
    def __init__(self):
        """
        It stops trainning when stop kernel is hit. The trainning stops at end of epoch.
        """
        super(KeyboardInterruptCallback, self).__init__()
        self.stopped_training = False

        # Register a signal handler for SIGINT (keyboard interrupt)
        signal.signal(signal.SIGINT, self.handle_signal)

    def handle_signal(self, signum, frame):
        print("\nTraining interrupted. Stopping training...")
        self.stopped_training = True

    def on_epoch_end(self, epoch, logs=None):
        if self.stopped_training:
            self.model.stop_training = True