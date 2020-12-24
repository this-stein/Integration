#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Description:   CCN Model Image Classification
#Authors:       Kevin Jordi, Sandro Bürgler, This Steinmetz
#Version:       2.0
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Import Modules
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import io
from tensorflow.keras import datasets, layers, models, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow import keras

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#extra solve hardware errors which let programm crash
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#variables & dataset
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
classes = ["Abfall", "Beleuchtung", "Brunnen", "Graffiti", "Gruenflaechen", "Schaedlinge", "Signalisation", "Strasse", "VBZ"]
img_height = 226
img_width = 205
epochs=1
colormode = "rgb" #"grayscale" or "rgb"
batchsize = 64
plotimages = False
datadir = "C:/bereinigt"

#value for ccn input layer depending on colormode
if colormode == "rgb":
  channelcount = 3
else:
  channelcount = 1


training = tf.keras.preprocessing.image_dataset_from_directory(
    datadir, 
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode=colormode,
    batch_size=batchsize,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training",
    interpolation="bilinear",
    follow_links=False
)
validation = tf.keras.preprocessing.image_dataset_from_directory(
    datadir,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode=colormode,
    image_size=(img_height, img_width),
    batch_size=batchsize,
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation",
    interpolation="bilinear",
    follow_links=False
)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#functions
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  test_pred = model.predict_classes(image_batch)

  con_mat = tf.math.confusion_matrix(labels=labels_batch, predictions=test_pred).numpy()
  con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

  con_mat_df = pd.DataFrame(con_mat_norm,
                     index = classes, 
                     columns = classes)

  figure = plt.figure(figsize=(8, 8))
 ma
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


#if Value is true --> 9 pictures will be plot with their categorie
if plotimages == True: 

  class_names = training.class_names
  print(class_names)
  plt.figure(figsize=(10, 10))
  for images, labels in training.take(1):
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
      plt.title(class_names[labels[i]])
      plt.axis("off")
  plt.show()


#Normalization layer for training data and validation data / data extension layer for training data

normalizationtraining_layer = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
  tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255)]) 

normalizationvalidation_layer = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255)])

#normalize and expand Training Data
normalized_ds = training.map(lambda x, y: (normalizationtraining_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
#normalize Validation Data
normalized_valdation = validation.map(lambda x, y: (normalizationvalidation_layer(x), y))
image_batch, labels_batch = next(iter(normalized_valdation))
# Check features (pixels values) are now in `[0,1]`,  if so dataset is normalized.
print(np.min(first_image), np.max(first_image)) 


# #if Value plotinages is true --> Image Plot of the data extension is displayed (horizontal Flip Zoom, rotation). 
# Attetion: normalization in normalizationtraining_layer must be commented out to display images last 
if plotimages == True: 
  plt.figure(figsize=(10, 10))
  for images, _ in normalized_ds.take(1):
    for i in range(9):
      augmented_images = normalizationtraining_layer(images)
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(augmented_images[0].numpy().astype("uint8"), cmap='gray')
      plt.axis("off")
  plt.show()


#model tunning loading data in memory

AUTOTUNE = tf.data.experimental.AUTOTUNE
training = training.cache().prefetch(buffer_size=AUTOTUNE)
validation = validation.cache().prefetch(buffer_size=AUTOTUNE)

#neuronales model

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(img_height, img_width, channelcount)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Dense(9))
model.add(Activation('softmax'))

model.summary()

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer='adam',
              metrics=['accuracy'])



#tensorboard visualisation of model.fit
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
file_writer = tf.summary.create_file_writer(log_dir + '/cm')

#training process of CNN
model.fit(
  normalized_ds,
  validation_data=normalized_valdation,
  epochs=epochs,
  shuffle=True,
  callbacks=[tensorboard_callback, cm_callback ]
)

#save the model that we can use it for other pictures
model.save('cnn-images-color-avoid-overfitting-bereinigt') 