import click
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Images pre-processing function
def preprocess_image(image):
    # Resize the image to the desired dimensions
    image = tf.image.resize(image, [256, 256])

    # Normalize the pixel values to the range [-1, 1]
    image = (tf.cast(image, tf.float32) / 127.5) - 1

    return image


@click.command()
@click.option("--model", required=True, help="Path to the saved model")
@click.option("--input-image", required=True, help="Path to the input image")
@click.option("--save-gen", is_flag=True, help="Save the generated image if provided")
def main(model, input_image, save_gen):
    # Load the saved model
    model = tf.keras.models.load_model(model)

    # Load and preprocess the input image
    original_image = tf.io.read_file(input_image)
    original_image = tf.image.decode_jpeg(original_image, channels=3)
    inference_image = preprocess_image(original_image)

    # Run inference
    generated_image = model(np.expand_dims(inference_image, axis=0)).numpy()

    # Denormalize the generated image
    generated_image = ((generated_image + 1) * 127.5).astype(np.uint8)
    generated_image = np.squeeze(generated_image, axis=0)

    # Visualize the result
    plt.figure(figsize=(10, 5))
    original_image = tf.image.resize(original_image, [256, 256])

    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(original_image.numpy().astype(np.uint8))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Generated Image")
    plt.imshow(generated_image)
    plt.axis("off")

    # Save the figure with input/generated image
    if save_gen:
        plt.savefig(input_image[:-4] + "_gen.png", transparent=True)

    plt.show()


if __name__ == "__main__":
    main()
