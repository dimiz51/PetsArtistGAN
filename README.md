# Datasets
In order to create the training and validation sets for our PetsGAN
model run follow the steps below.

### Step 1: Download and generate the Oxford Pets dataset from Tensorflow-Datasets
```
python ./code/get_pets_ds.py
```

### Step 2: Create the Art styles dataset as you see fit:
You can use any of the three provided dataset generators under *./code/art_dataset.ipynb* to create the art styles dataset.

**Art dataset source**: https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time


### Step 3: Merge the datasets and prepare the PetsGAN_train dataset:

1. **Create the New Dataset Directory:**
   ```bash
   mkdir datasets/PetsGAN_train
   ```

2. **Copy art dataset images:**
   ```bash 
   cp -r datasets/art_dataset/train/* datasets/PetsGAN_train/
   ```

3. **Copy pet dataset images:**
   ```bash
   cp -r datasets/oxford_pets/train/*/* datasets/PetsGAN_train/
   ```

Now your newly created training dataset should look like this:

```
datasets/PetsGAN_train/ 
├── art_dataset 
│   └── train 
│       ├── art_image1.jpg
│       ├── art_image2.jpg 
│       └── ... 
└── oxford_pets 
    └─ train 
        ├── Abyssinian 
        │   ├── Abyssinian_1.jpg 
        │   ├── Abyssinian_2.jpg 
        │   └── ... 
        ├── Bengal 
        │   ├── Bengal_1.jpg 
        │   ├── Bengal_2.jpg 
        │   └── ... 
        ├── ... 

```