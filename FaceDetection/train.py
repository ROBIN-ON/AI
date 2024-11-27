import pylab as pl
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Using the TensorFlow library's tf.data, the given code loads two datasets, labeled "train" and "validation."Dataset.load()
# is a function. For training and verifying machine learning models, these datasets are frequently used. Examples for training
# make up the train_ds dataset, whereas examples for validation make up the val_ds dataset.

train_ds = tf.data.Dataset.load('datacollect/train')
val_ds = tf.data.Dataset.load('datacollect/validation')


# A list of names or labels for various classes or categories is represented in the provided code by the variable class_names.
# A distinct class is associated with each entry in the list. The total number of classes in the classification task is stored in the num_classes variable
# in this case being 4, which is a total of 4.

class_names = ['dawa','kripa' , 'kritika', 'prasanna', 'robin']
num_classes = 5

 # A TensorFlow Keras Sequential model specifically designed for picture categorization is defined
# using the given code. The output is flattened after a series of convolutional layers with maximum pooling are used.
# The flattened data is then sent via two completely connected layers, one of which is enabled using ReLU and the other not.
# The number of classification classes is accommodated in the top layer.


model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)

])

# The 'adam' optimizer, the Sparse Categorical Crossentropy loss function, and accuracy as the evaluation
# metric are all specified in this code, which sets up the construction of a TensorFlow model. As a result,
# the model is ready for training, specifying crucial components including optimization, goal function, and performance measures.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# start the process of model training.
model.fit(train_ds, epochs=15)

model.save('model/my_model')