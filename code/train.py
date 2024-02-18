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

IMG_HEIGHT = 256
IMG_WIDTH = 256

# Images pre-processing function
def preprocess_image(image):
    # Resize the image to the desired dimensions
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize the pixel values to the range [-1, 1]
    image = (tf.cast(image, tf.float32) / 127.5) - 1

    return image

# Wrapper function for loading/pre-processing
def load_and_preprocess_image(image_path):
    # Read and decode image file
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Preprocess image
    image = preprocess_image(image)

    return image


# Create training dataloaders
def create_training_dataloader(dataset_dir: str, batch_size: int):

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

    # Map preprocessing function to each image
    art_dataset = art_dataset.map(load_and_preprocess_image)
    pets_dataset = pets_dataset.map(load_and_preprocess_image)

    # Re-shuffle and batch the datasets
    art_dataset = art_dataset.batch(batch_size).shuffle(buffer_size=batch_size * 4)
    pets_dataset = pets_dataset.batch(batch_size).shuffle(buffer_size=batch_size * 4)

    return pets_dataset, art_dataset


def main(epochs: int, batch_size: int):
    # Create training dataloader
    pets_train, art_train = create_training_dataloader(
        "../datasets/PetsGAN_train", batch_size
    )

    print(f"Creating PetsGAN model...")

    # Create and compile model
    model = CycleGAN()
    model.compile()

    # Test in/out
    model(np.random.rand(1, 256, 256, 3))

    # Print summary
    model.summary()

    # Train the model
    print(f"Starting training...")
    for epoch in range(epochs):
        for image_x, image_y in tf.data.Dataset.zip((pets_train, art_train)):
            model.train_step(image_x, image_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs (default: 1)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size (default: 4)"
    )
    args = parser.parse_args()
    main(args.epochs, args.batch_size)
