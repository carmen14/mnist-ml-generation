"""
Main script for training and inference of:
    - DCGAN (Deep Convolutional GAN)
    - VAE (Variational Autoencoder)
    - DDPM (Denoising Diffusion Probabilistic Model)

Dataset: MNIST, resized to 32x32 and normalized to [-1, 1]
"""

import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from models.dcgan_ml_model_tf import DCGAN
from models.vae_ml_model_tf import VAE
from models.ddpm_ml_model_tf import SimpleDDPM

def load_dataset(batch_size: int):
    """Load and preprocess MNIST dataset."""
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_train = tf.image.resize(x_train, [32, 32])  # resize to 32x32
    x_train = (x_train / 127.5) - 1.0  # normalize to [-1, 1]
    return tf.data.Dataset.from_tensor_slices(x_train).shuffle(len(x_train)).batch(batch_size)

def train(models, dataset, epochs: int):
    """Train DCGAN, VAE, and DDPM for a specified number of epochs."""
    dcgan, vae, ddpm = models

    for epoch in range(epochs):
        for batch in dataset:
            batch = tf.cast(batch, tf.float32)

            vae_loss = vae.train_step(batch)
            ddpm_loss = ddpm.train_step(batch)
            gen_loss, disc_loss = dcgan.train_step(batch)

        print(f"Epoch {epoch+1}/{epochs} | "f"DCGAN gen_loss={gen_loss.numpy():.4f}, disc_loss={disc_loss.numpy():.4f} | "f"VAE loss={vae_loss.numpy():.4f} | "f"DDPM loss={ddpm_loss.numpy():.4f}")

def generate_comparison_plot(x_sample, dcgan_imgs, vae_out, ddpm_out, epochs, output_path="output/dcgan_vae_ddpm_ml_model_comparison.png"):
    """Generate a publication-quality comparison plot of DCGAN, VAE, and DDPM models."""
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs = axs.flatten()
    titles = ["Original", "DCGAN (Deep Convolutional GAN)", "VAE (Variational Autoencoder)", "DDPM (Denoising Diffusion\nProbabilistic Model)"]
    image_sets = [x_sample[:16], dcgan_imgs, vae_out, ddpm_out]
    
    for ax, imgs, title in zip(axs, image_sets, titles):
        imgs = (imgs + 1) / 2.0  # normalize to [0, 1] for imshow
        
        # Build 4x4 grid
        rows = []
        for r in range(4):
            row = tf.concat([imgs[c + r*4] for c in range(4)], axis=1)
            rows.append(row)
        grid = tf.concat(rows, axis=0)
        
        ax.imshow(grid.numpy().squeeze(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    
    fig.suptitle(f"Comparison of Generative Models (for {epochs} epochs)", fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison plot to {output_path}")

def generate_images(models, dataset, z_dim: int):
    """Generate samples from all models."""
    dcgan, vae, ddpm = models

    # DCGAN generation
    z = tf.random.normal([16, z_dim])
    dcgan_imgs = dcgan(z)

    # VAE reconstruction
    x_sample = next(iter(dataset))
    vae_out, _, _ = vae(x_sample[:16])

    # DDPM denoising
    noise = tf.random.normal(shape=x_sample[:16].shape)
    ddpm_out = ddpm(noise)
    
    return x_sample, dcgan_imgs, vae_out, ddpm_out

def main(args):

    print("\nLoading dataset...")
    train_dataset = load_dataset(args.batch_size)

    print("Initializing models...")
    dcgan_model = DCGAN(z_dim=args.z_dim)
    vae_model   = VAE(latent_dim=64)
    ddpm_model  = SimpleDDPM()

    print("Starting training...\n")
    train((dcgan_model, vae_model, ddpm_model), train_dataset, args.epochs)

    print("\nGenerating images and plotting results...")
    x_sample, dcgan_imgs, vae_out, ddpm_out = generate_images((dcgan_model, vae_model, ddpm_model), train_dataset, args.z_dim)
    generate_comparison_plot(x_sample, dcgan_imgs, vae_out, ddpm_out, args.epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate DCGAN, VAE, and DDPM models on MNIST.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--z_dim", type=int, default=100, help="Latent vector size for DCGAN")
    args = parser.parse_args()

    main(args)
