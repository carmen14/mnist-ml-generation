"""VAE (Variational Autoencoder) implementation using TensorFlow/Keras."""

import tensorflow as tf
from tensorflow.keras import layers, Model

# VAE Class
class VAE(Model):
    """Variational Autoencoder (VAE) model.

    Args:
        latent_dim (int): Dimension of the latent space.
        img_shape (tuple): Shape of input images (height, width, channels).
    """
    
    def __init__(self, latent_dim=20, img_shape=(32, 32, 1)): # If RGB: img_shape=(32, 32, 3)
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        
        # Encoder
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(400, activation='relu')
        self.fc_mu_logvar = layers.Dense(2*latent_dim)
        
        # Decoder
        self.fc_dec1 = layers.Dense(400, activation='relu')
        self.fc_dec2 = layers.Dense(img_shape[0]*img_shape[1]*img_shape[2], activation='tanh')
        self.reshape = layers.Reshape(img_shape)
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
    
    def encode(self, x):
        """Encode input images into latent mean and log-variance.

        Args:
            x (Tensor): Input images.

        Returns:
            Tuple[Tensor, Tensor]: Mean and log-variance of latent distribution.
        """
        x = self.flatten(x)
        h = self.fc1(x)
        mu_logvar = self.fc_mu_logvar(h)
        mu, logvar = tf.split(mu_logvar, num_or_size_splits=2, axis=1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Sample latent vector using the reparameterization trick.

        Args:
            mu (Tensor): Latent mean.
            logvar (Tensor): Latent log-variance.

        Returns:
            Tensor: Sampled latent vector.
        """
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5*logvar)*eps
    
    def decode(self, z):
        """Decode latent vector back to reconstructed image.

        Args:
            z (Tensor): Latent vector.

        Returns:
            Tensor: Reconstructed images.
        """
        x = self.fc_dec1(z)
        x = self.fc_dec2(x)
        x = self.reshape(x)
        return x
    
    def call(self, x):
        """Forward pass through VAE: encode → reparameterize → decode.

        Args:
            x (Tensor): Input images.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Reconstructed images, latent mean, latent log-variance.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def compute_loss(self, x, x_recon, mu, logvar):
        """
        Computes the total loss for the Variational Autoencoder (VAE).

        Args:
            x (tf.Tensor): Original input images.
            x_recon (tf.Tensor): Reconstructed images produced by the VAE.
            mu (tf.Tensor): Mean of the latent distribution.
            logvar (tf.Tensor): Log-variance of the latent distribution.

        Returns:
            tf.Tensor: Scalar tensor representing the sum of the reconstruction loss (MSE or binary crossentropy) and KL divergence.
        """
        recon_loss = tf.reduce_mean(tf.square(x - x_recon))        
        kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
        
        return recon_loss + kl_loss
    
    @tf.function
    def train_step(self, x):
        """
        Performs a single training step for the Variational Autoencoder (VAE).

        Args:
            x (tf.Tensor): A batch of input images.

        Returns:
            tf.Tensor: The computed VAE loss for the batch, combining reconstruction and KL divergence.
        """
        with tf.GradientTape() as tape:
            x_recon, mu, logvar = self(x)
            loss = self.compute_loss(x, x_recon, mu, logvar)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss
