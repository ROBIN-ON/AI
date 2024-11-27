import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Four variables that are frequently utilized in image processing activities are given specific values by this bit of code.
# The batch_size variable, which specifies the quantity of images that will be processed collectively in a single batch during training or inference,
# is set to 32. The required image dimensions in pixels are represented by the img_height and img_width variables, which are both set to 180.

batch_size = 32
img_height = 180
img_width = 180
image_directory = 'output-image'

train_ds = tf.keras.utils.image_dataset_from_directory(
    # the name of the directory where the photos are.
    image_directory,
    validation_split=0.2,

    # Determines whether a dataset should be created for validation or
    # for training. In this instance, the word "training" is used to specify
    # that the function will provide a dataset for training.
    subset="training",
    # a starting point for randomizing the dataset. This makes sure that the
    # dataset is consistently randomized each time the code is executed with the same seed.
    seed=123,
    image_size=(img_height, img_width),

    # how many pictures should be included in each dataset batch.
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    # the directory's path where the dataset of images is located.
    image_directory,

    # The portion of the dataset to be used for validation is specified by this option.
    # In this scenario, 20% of the dataset will be set aside for validation, while the
    # remaining 80% will be used for training.
    validation_split=0.2,
    subset="validation",


    # For reproducibility, this parameter determines the random seed. It makes that the dataset splitting
    # is uniform across many runs. The intended size for the input photos is specified by this option.
    # It guarantees that every image is scaled to the required height and width.The number of photos in each
    # batch of the dataset is determined by this parameter. It influences how finely the model is updated during training.
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# This code snippet generates a normalization layer and applies it to the input photos of the training dataset, guaranteeing that
# the pixel values are normalized between 0 and 1. This normalization phase is frequently carried out to enhance the convergence and stability
# of deep learning models during training.

normalization_layer = tf.keras.layers.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE

# Using a memory cache to store the supplied data, this method aims to improve training efficiency.
# Caching entails keeping the dataset in a fast memory location, like RAM, to provide quick access during training.
# The method avoids having to frequently load the disk, which can be a time-consuming process, by caching the dataset.
# Iterations of training become more effective as a result of the caching strategy's dramatic reduction in data retrieval time.

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# It appears that train_ds is referring to a dataset object or data structure that holds training data. The dataset must
# be saved to a certain location using the save function, which is probably a method or function connected to this dataset object.
train_ds.save('datacollect/train')

# 'datacollect/validation', a relative file path, is the destination that has been supplied. The validation dataset will
# therefore be saved with the filename or identifier "validation" in the directory "datacollect."
val_ds.save('datacollect/validation')