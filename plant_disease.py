from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import cv2
import json
# Utility
import tensorflow.compat.v1.keras.backend as K
from PIL import Image
import random
from collections import Counter
import tensorflow_hub as hub
import os
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import matplotlib.pylab as plt
import numpy as np
import os
import time
from os.path import exists

os.environ['KMP_DUPLICATE_LIB_OK']='True'
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

data_dir = os.path.join("D:\program\MLmodel\content\drive\dataset", 'dataset')
train_dir = os.path.join("D:\program\MLmodel\content\drive\dataset", 'train')
validation_dir = os.path.join("D:\program\MLmodel\content\drive\dataset", 'valid')
test_dir = os.path.join("D:\program\MLmodel\content\drive\dataset",'test')

def count(dir, counter=0):
    "returns number of files in dir and subdirs"
    for pack in os.walk(dir):
        for f in pack[2]:
            counter += 1
    return dir + " : " + str(counter) + "files"


# print('total images for training :', count(train_dir))
# print('total images for validation :', count(validation_dir))
# print('total images for TEST :', count(test_dir))

# with open('D:\program\MLmodel\content\drive\categories.json', 'r') as f:
#     categories = json.load(f)
#     classes = list(categories.values())
# print (classes)

# print('Number of classes:',len(classes))

IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 64 #@param {type:"integer"}

valid_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = valid_data.flow_from_directory(
    validation_dir, 
    shuffle=False, 
    seed=42,
    color_mode="rgb", 
    class_mode="categorical",
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE)

do_data_augmentation = True 
if do_data_augmentation:
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale = 1./255,
      rotation_range=40,
      horizontal_flip=True,
      width_shift_range=0.2, 
      height_shift_range=0.2,
      shear_range=0.2, 
      zoom_range=0.2,
      fill_mode='nearest' )
else:
  train_datagen = valid_data
  
train_generator = train_datagen.flow_from_directory(
    train_dir, 
    subset="training", 
    shuffle=True, 
    seed=42,
    color_mode="rgb", 
    class_mode="categorical",
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE)

test_data =tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_data.flow_from_directory(
    test_dir, 
    shuffle=False, 
    seed=42,
    color_mode="rgb", 
    class_mode="categorical",
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE)


# To see iof the model already exists
from pathlib import Path
my_file = Path("D:\program\MLmodel\my_model\saved_model.pb")
if(my_file.exists()):
  print("Yes! It exist")
  model= tf.keras.models.load_model('D:\program\MLmodel\my_model')
  loss,acc= model.evaluate(validation_generator, verbose=2)
  
else:
  layer = hub.KerasLayer("D:\program\MLmodel\SFtvector", 
                 output_shape=[1280],trainable=False)
  model = tf.keras.Sequential([
      layer,
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dropout(rate=0.2),
  tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])
#Compile model specifying the optimizer learning rate
  LEARNING_RATE = 0.001 
  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
    loss='categorical_crossentropy',
    metrics=['accuracy'])
  EPOCHS=10 #@param {type:"integer"}
  history = model.fit(
          train_generator,

          steps_per_epoch=train_generator.samples//train_generator.batch_size,
          epochs=EPOCHS,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples//validation_generator.batch_size)
try:
  # print(len(history))
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(EPOCHS)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')
  plt.ylabel("Accuracy (training and validation)")
  plt.xlabel("Training Steps")

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.ylabel("Loss (training and validation)")
  plt.xlabel("Training Steps")
  plt.show()
except NameError:
  print("No variable history")
def load_image(filename):
    # img = cv2.imread(os.path.join(data_dir, validation_dir, filename))
    img = cv2.imread("D:\program\MLmodel\input.jpg")
    # print("D:\program\MLmodel\"+f'{filename}'")
    # print(img)
    img = cv2.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]) )
    img = img /255
    return img


def predict(image):
    probabilities = model.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    return {classes[class_idx]: probabilities[class_idx]}

def predict_reload(image):
    probabilities = model.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}


# # convert the model to TFLite
# # !mkdir "tflite_models"
# directory = "plant_disease_model.tflite"
# # Parent Directory path
# parent_dir = "D:/program/Mlmodel/tflite_models/"
# # Path
# path = os.path.join(parent_dir, directory)
# os.mkdir(path)
# tflite_models = "tflite_models/plant_disease_model.tflite"
# # Get the concrete function from the Keras model.
# run_model = tf.function(lambda x : model(x))
# # Save the concrete function.
# concrete_func = run_model.get_concrete_function(
#     tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
# )


# # Convert the model to standard TensorFlow Lite model
# converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.allow_custom_ops=True
# tflite_model = converter.convert()
# #directly used keras model to conver 
# tflite_model = converter.convert()

# with open('modelOne.tflite', 'wb') as f:
#   f.write(tflite_model)

# # Print the signatures from the converted model
# interpreter = tf.lite.Interpreter(model_content=tflite_model)
# # interpreter.allocate_tensors()
# # signatures = interpreter.get_signature_list()
# print("input details: ",interpreter.get_input_details()[:])
# print("output details: ",interpreter.get_output_details()[:])
