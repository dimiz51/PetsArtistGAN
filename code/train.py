# Set the environment variable to suppress TensorFlow logging
import sys

sys.path.append("./")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import random
from model import CycleGAN
import numpy as np
import argparse
from functools import partial
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt


IMG_HEIGHT = 256
IMG_WIDTH = 256


def augmentations(image):
    """Augmentations function for the training set"""
    # Apply random horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Apply random rotation
    image = tf.image.rot90(image, k=random.randint(0, 3))

    # Apply color augmentation
    image = tf.image.random_brightness(image, max_delta=0.2)

    return image


# Images pre-processing function
def preprocess_image(image, augment=False):

    # Augmentations for training set
    if augment:
        image = augmentations(image)

    # Resize the image to the desired dimensions
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize the pixel values to the range [-1, 1]
    image = (tf.cast(image, tf.float32) / 127.5) - 1

    return image


# Wrapper function for loading/pre-processing
def load_and_preprocess_image(image_path, augment=False):
    # Read and decode image file
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Preprocess image
    image = preprocess_image(image, augment)

    return image


# Wrapper function for pre-processing with augmentation for training
pre_process_augment = partial(load_and_preprocess_image, augment=True)

# Create training dataloaders
def create_training_dataloader(dataset_dir: str, batch_size: int):
    """Training dataset dataloader"""
    # List all image files
    art_files = tf.io.gfile.glob(os.path.join(dataset_dir, "art", "*.jpg"))
    pets_files = tf.io.gfile.glob(os.path.join(dataset_dir, "pets", "*.jpg"))

    # Remove half of the files for testing
    art_files = art_files[: int(len(art_files) // 25)]
    pets_files = pets_files[: int(len(pets_files) // 25)]

    # Shuffle the file paths
    random.shuffle(art_files)
    random.shuffle(pets_files)

    # Match the dataset sizes by removing excess files
    if len(art_files) > len(pets_files):
        art_files = art_files[: len(pets_files)]
    elif len(pets_files) > len(art_files):
        pets_files = pets_files[: len(art_files)]

    print(
        f"Creating dataset from: {len(pets_files)} pet and {len(art_files)} art images.."
    )
    print(f"Batch size: {batch_size}")
    print(f"Resized image size: {IMG_WIDTH}x{IMG_HEIGHT}x3")
    print(
        f"Total number of training batches: {int(min(len(pets_files), len(art_files)) / batch_size)}"
    )

    # Create TensorFlow dataset from the file paths
    art_dataset = tf.data.Dataset.from_tensor_slices(art_files)
    pets_dataset = tf.data.Dataset.from_tensor_slices(pets_files)

    # Map preprocessing function to each image with augmentation and parallel calls
    art_dataset = art_dataset.map(
        pre_process_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    pets_dataset = pets_dataset.map(
        pre_process_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Re-shuffle and batch the datasets
    art_dataset = (
        art_dataset.shuffle(buffer_size=len(art_files))
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    pets_dataset = (
        pets_dataset.shuffle(buffer_size=len(pets_files))
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    return pets_dataset, art_dataset


def create_validation_dataloader(dataset_dir: str, batch_size: int):
    """Validation dataset dataloader"""
    # List all image files
    art_files = tf.io.gfile.glob(os.path.join(dataset_dir, "art", "*.jpg"))
    pets_files = tf.io.gfile.glob(os.path.join(dataset_dir, "pets", "*.jpg"))

    # Shuffle the file paths
    random.shuffle(art_files)
    random.shuffle(pets_files)

    # Match the dataset sizes by removing excess files
    min_length = min(len(art_files), len(pets_files))
    art_files = art_files[:min_length]
    pets_files = pets_files[:min_length]

    print(
        f"Creating validation dataset from: {len(pets_files)} pet and {len(art_files)} art images.."
    )
    print(f"Batch size: {batch_size}")
    print(f"Resized image size: {IMG_WIDTH}x{IMG_HEIGHT}x3")
    print(
        f"Total number of validation batches: {int(min(len(pets_files), len(art_files)) / batch_size)}"
    )

    # Create TensorFlow dataset from the file paths
    art_dataset = tf.data.Dataset.from_tensor_slices(art_files)
    pets_dataset = tf.data.Dataset.from_tensor_slices(pets_files)

    # Map preprocessing function to each image without augmentation and parallel calls
    art_dataset = art_dataset.map(
        load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    pets_dataset = pets_dataset.map(
        load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Batch the datasets
    art_dataset = art_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    pets_dataset = pets_dataset.batch(batch_size).prefetch(
        tf.data.experimental.AUTOTUNE
    )

    return pets_dataset, art_dataset


# -------------------Training Callbacks------------
class VisualizationCallback(Callback):
    def __init__(self, input_image: tf.Tensor):
        self.input_image = input_image

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            # Run inference
            generated_image = self.model(self.input_image).numpy()

            # Denormalize the generated image
            generated_image = ((generated_image + 1) * 127.5).astype(np.uint8)
            generated_image = np.squeeze(generated_image, axis=0)
            print(f"\nTesting generator on epoch {epoch}...")
            # Visualize the result
            plt.figure(figsize=(10, 5))
            plt.imshow(generated_image)
            plt.title(f"Generated Image (Epoch {epoch})")
            plt.axis("off")
            plt.show()


class LinearAnnealingScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, start_epoch, final_epoch):
        super(LinearAnnealingScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.start_epoch = start_epoch
        self.final_epoch = final_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.start_epoch and epoch <= self.final_epoch:
            current_lr = self.initial_lr * (
                1 - (epoch - self.start_epoch) / (self.final_epoch - self.start_epoch)
            )

            # Update learning rate of each optimizer in the model
            for optimizer in [
                self.model.gen_G_optimizer,
                self.model.gen_F_optimizer,
                self.model.disc_X_optimizer,
                self.model.disc_Y_optimizer,
            ]:
                tf.keras.backend.set_value(optimizer.learning_rate, current_lr)
            print(f"Epoch {epoch + 1}: Setting learning rate to {current_lr}")


def main(epochs: int, batch_size: int, constant_lr: int):
    # Load training data
    pets_train, art_train = create_training_dataloader(
        "../datasets/PetsGAN_train", batch_size
    )

    # Load validation data
    pets_val, art_val = create_validation_dataloader(
        "../datasets/PetsGAN_validation", batch_size
    )

    # Take an image from the validation set for the visualization callback
    callback_image = next(iter(pets_val))

    print(f"Creating PetsArtistGAN model...")

    # Create and compile model
    model = CycleGAN()
    model.compile()

    # Test in/out
    model(np.random.rand(1, 256, 256, 3))

    # Print summary
    model.summary()

    # Training callback
    visualize_callback = VisualizationCallback(callback_image)
    lr_scheduler = LinearAnnealingScheduler(
        initial_lr=2e-4, start_epoch=constant_lr, final_epoch=epochs
    )

    # Train the model
    print(f"Starting training...")
    history = model.fit(
        tf.data.Dataset.zip((pets_train, art_train)),
        epochs=epochs,
        validation_data=tf.data.Dataset.zip((pets_val, art_val)),
        callbacks=[visualize_callback, lr_scheduler],
    )

    # Save model and weights after training
    model.save("../PetsGAN")
    model.save_weights(f"../PetsGAN_{epochs}_{batch_size}.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Default: 150",
    )
    parser.add_argument(
        "--constant_lr",
        type=int,
        default=70,
        help="Epochs count to keep constant lr | Default: 70",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Original paper: 1")
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.constant_lr)
