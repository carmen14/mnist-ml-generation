"""DCGAN (Deep Convolutional GAN) implementation using TensorFlow/Keras."""

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses

class DCGAN(Model):
    """Deep Convolutional Generative Adversarial Network.

    Combines a generator and discriminator for image synthesis.

    Args:
        z_dim (int): Dimensionality of latent vector input to generator.
        img_shape (tuple): Shape of the generated/real images (height, width, channels).
    """
    
    def __init__(self, z_dim=100, img_shape=(32, 32, 1)): # If RGB: img_shape=(32, 32, 3)
        super().__init__()
        self.z_dim = z_dim
        self.img_shape = img_shape

        # Generator
        self.generator = tf.keras.Sequential([
            layers.Input(shape=(z_dim,)),
            layers.Dense(4*4*256, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Reshape((4, 4, 256)),
            
            layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(img_shape[2], 4, strides=2, padding='same', activation='tanh')
        ])

        # Discriminator
        self.discriminator = tf.keras.Sequential([
            layers.Input(shape=img_shape),
            layers.Conv2D(64, 4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            
            layers.Conv2D(128, 4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])

        # Optimizers
        self.gen_optimizer = optimizers.Adam(1e-4)
        self.disc_optimizer = optimizers.Adam(1e-4)

        # Loss
        self.cross_entropy = losses.BinaryCrossentropy(from_logits=False)

    # Generator loss
    def generator_loss(self, fake_output):
        """Compute generator loss.

        Args:
            fake_output (Tensor): Discriminator predictions on generated images.

        Returns:
            Tensor: Generator loss.
        """
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    # Discriminator loss
    def discriminator_loss(self, real_output, fake_output):
        """Compute discriminator loss.

        Args:
            real_output (Tensor): Discriminator predictions on real images.
            fake_output (Tensor): Discriminator predictions on generated images.

        Returns:
            Tensor: Discriminator loss.
        """
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    # Forward: generate images
    def call(self, z):
        """Forward pass through the generator.

        Args:
            z (Tensor): Random latent vectors.

        Returns:
            Tensor: Generated images.
        """
        return self.generator(z)

    # Training step
    @tf.function
    def train_step(self, real_images):
        """Perform one training step for generator and discriminator.

        Args:
            real_images (Tensor): Batch of real images.

        Returns:
            Tuple[Tensor, Tensor]: Generator loss and discriminator loss.
        """
        z = tf.random.normal([tf.shape(real_images)[0], self.z_dim])

        with tf.GradientTape(persistent=True) as tape:
            fake_images = self.generator(z, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        grads_gen = tape.gradient(gen_loss, self.generator.trainable_variables)
        grads_disc = tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(grads_gen, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(grads_disc, self.discriminator.trainable_variables))

        return gen_loss, disc_loss
