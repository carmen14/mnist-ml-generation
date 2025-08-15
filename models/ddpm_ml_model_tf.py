"""Simple DDPM (Denoising Diffusion Probabilistic Model) implementation using TensorFlow/Keras."""

import tensorflow as tf
from tensorflow.keras import layers, Model

# SimpleDDPM Class
class SimpleDDPM(Model):
    """Simple UNet-like denoiser for DDPM.

    Args:
        img_shape (tuple): Shape of input images (height, width, channels).
    """
    
    def __init__(self, img_shape=(32, 32, 1)): # If RGB: img_shape=(32, 32, 3)
        super().__init__()
        self.conv1 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(img_shape[2], 3, padding='same')
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
    
    def call(self, x):
        """Forward pass through the DDPM denoiser.

        Args:
            x (Tensor): Noisy input images.

        Returns:
            Tensor: Denoised images.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    
    @tf.function
    def train_step(self, x):
        """
        Performs a single training step for the Denoising Diffusion Probabilistic Model (DDPM).

        Args:
            x (tf.Tensor): A batch of input images.

        Returns:
            tf.Tensor: The computed mean squared error loss between the predicted noise and the added noise.
        """
        noise = tf.random.normal(shape=x.shape)
        with tf.GradientTape() as tape:
            pred_noise = self(x + noise)
            loss = tf.reduce_mean(tf.square(pred_noise - noise))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

# Loss
def ddpm_loss(pred, target):
    """Compute mean squared error loss for DDPM.

    Args:
        pred (Tensor): Predicted images.
        target (Tensor): Ground-truth images.

    Returns:
        Tensor: Loss value.
    """
    return tf.reduce_mean(tf.keras.losses.mse(pred, target))
