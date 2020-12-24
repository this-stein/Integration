#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Description:   CCN Model variable Image Size
#Authors:       Kevin Jordi, Sandro Bürgler, This Steinmetz
#Version:       2.0
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Import Modules
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import datetime
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Sequential
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import pathlib
import numpy as np
import io
import pandas as pd
import seaborn as sns

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#extra solve hardware errors which let programm crash
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Variables
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

list_ds = tf.data.Dataset.list_files(str('C:/bereinigt/*.jpeg'), shuffle=True)

classes = ["Abfall", "Beleuchtung", "Brunnen", "Graffiti", "Gruenflaechen", "Schaedlinge", "Signalisation", "Strasse", "VBZ"]
epochs =1800
validationsplit = 0.1


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#functions
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == classes
  # Integer encode the label
  return tf.argmax(one_hot)

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=10000)
  ds = ds.batch(1)
  return ds


def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  test_pred = model.predict_classes(image_batch)

  con_mat = tf.math.confusion_matrix(labels=labels_batch, predictions=test_pred).numpy()
  con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

  con_mat_df = pd.DataFrame(con_mat_norm,
                     index = classes, 
                     columns = classes)

  figure = plt.figure(figsize=(8, 8))
  sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  
  buf = io.BytesIO()
  plt.savefig(buf, format='png')

  plt.close(figure)
  buf.seek(0)
  image = tf.image.decode_png(buf.getvalue(), channels=4)

  image = tf.expand_dims(image, 0)
  
  # Log the confusion matrix as an image summary.
  with file_writer.as_default():
    tf.summary.image("Confusion Matrix", image, step=epoch)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Code
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
for f in list_ds.take(5):
  print(f.numpy())


#print(class_names)
image_count = len(list(list_ds))
print(image_count)
val_size = int(image_count * validationsplit) #split training and validation data
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)


##print(tf.data.experimental.cardinality(train_ds).numpy())
##print(tf.data.experimental.cardinality(val_ds).numpy())


#map the pictures with lable
training = train_ds.map(process_path)
validation = val_ds.map(process_path)



training = configure_for_performance(training)
validation = configure_for_performance(validation)



#Data normalization and data augmentation for training data
normalizationtraining_layer = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
  tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255)])

normalizationvalidation_layer = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255)])




normalized_ds = training.map(lambda x, y: (normalizationtraining_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
first_label = labels_batch[0]
#normalize ValData
normalized_valdation = validation.map(lambda x, y: (normalizationvalidation_layer(x), y))
image_batch, labels_batch = next(iter(normalized_valdation))
# Check features (pixels values) are now in `[0,1]`. equals check if pictures are normalized
print(np.min(first_image), np.max(first_image)) 





#neuronales model

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(None, None, 3))) #no fixed size equals none
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(GlobalAveragePooling2D())  
model.add(Dense(64))
model.add(Dense(9))
model.add(Activation('softmax'))


model.summary()

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer='adam',
              metrics=['accuracy'])



#tensorboard visualisation of model.fit
log_dir = "C:/TEMP/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#tensorboard visualisation of confusion matrix per epoch
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
file_writer = tf.summary.create_file_writer(log_dir + '/cm')


#fit model
model.fit(
  normalized_ds,
  validation_data=normalized_valdation,
  epochs=epochs,
  shuffle=True,
  callbacks=[tensorboard_callback, cm_callback]
)


#save model to use it on own data
model.save('cnn-varimages-newlayer') 

