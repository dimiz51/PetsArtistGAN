import tensorflow as tf
import tensorflow_datasets as tfds
import click
import os
from PIL import Image
import numpy as np


def create_label_mapping(info, inverse = False):
    label_mapping = {}
    for index, label_name in enumerate(info.features['label'].names):
        label_mapping[label_name] = index
    if inverse:
        inverse_mapping = {v: k for k, v in label_mapping.items()}
        return inverse_mapping
    return label_mapping

@click.command()
@click.option('--destination_dir', default='../datasets/oxford_pets', help='Destination directory to save the dataset')
def download_oxford_pets(destination_dir):
    # Load the Oxford pets
    (train_ds, test_ds, val_ds ), info = tfds.load(
        'oxford_iiit_pet:3.*.*',
        split=['train', 'test[:80%]', 'test[80%:100%]'],
        with_info=True)

    # Access and print dataset information
    print("Oxford pets dataset information:")
    print(f"Number of classes: {info.features['label'].num_classes}")
    print(f"Class names: {info.features['label'].names}")
    print(f"Features: {info.features}")
    print(f"Number of training examples: {info.splits['train'].num_examples}")
    print(f"Number of test examples: {int(info.splits['test'].num_examples * 0.8)}")
    print(f"Number of validation examples: {int(info.splits['test'].num_examples * 0.2)}")
    print(f"Dataset splits: {list(info.splits.keys())}")
    print(f"Dataset description: {info.description}")


    # Create directories for each split and label
    if os.path.exists(destination_dir):
        raise FileExistsError(f"Destination directory '{destination_dir}' already exists.")
    os.makedirs(destination_dir, exist_ok=False)
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(destination_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for label_name in info.features['label'].names:
            label_dir = os.path.join(split_dir, label_name)
            os.makedirs(label_dir, exist_ok=True)

    # Create class mapping between labels and idxs
    class_mapping = create_label_mapping(info, inverse= True)

    # Save the dataset on disk
    datasets = [train_ds, val_ds, test_ds]
    splits = ['train', 'validation', 'test']

    print('Saving image dataset on disk....')
    for split_idx, split_dataset in enumerate(datasets):
        iterator = iter(split_dataset)
        for i in range(0, len(split_dataset)):    
            # Read image
            image_data = next(iterator)
            image = image_data['image']
            class_idx = image_data['label']
            class_label = class_mapping[class_idx.numpy()]
            filename = image_data['file_name'].numpy().decode('utf-8')
            
            # Find destination path
            split_path = os.path.join(destination_dir, splits[split_idx])
            class_path = os.path.join(split_path, class_label)
            write_path = os.path.join(class_path,filename)

            # Save image
            image_array = np.array(image)
            image_pil = Image.fromarray(image_array)
            image_pil.save(write_path)

    print("Dataset downloaded and saved successfully!")

if __name__ == "__main__":
    download_oxford_pets()