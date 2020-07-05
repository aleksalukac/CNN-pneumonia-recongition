# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 22:41:26 2020

@author: Aleksa
"""
import matplotlib.pyplot as plt

from glob import glob
import numpy as np
import os
import PIL
import pandas as pd


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential

from keras.datasets import mnist
from keras.utils import to_categorical

from IPython.display import Image
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
    

# %%
def history_plot(hist, metric="acc", name="accuracy"):
    train_loss = hist.history["loss"]
    valid_loss = hist.history["val_loss"]
    train_metr = hist.history[metric]
    valid_metr = hist.history["val_"+metric]
    epochs = range(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_metr, 'bo', label='Training ' + name)
    plt.plot(epochs, valid_metr, 'k', label='Validation ' + name)
    plt.title('Training and validation ' + name)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, valid_loss, 'k', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    
# %% load labels

data = pd.read_csv(".\metadata\chest_xray_metadata.csv")

# %% load all images

images = []
label = []
data_small = []

allfiles = list(set().union(glob('*.jpg'), glob('*.jpeg')))

for filename in allfiles:
    im = image.load_img(filename, target_size=(150, 150))
    img_arr = image.img_to_array(im)
    
    image_label = data[data["X_ray_image_name"] == filename]["Label"]
    
    if(len(image_label) != 0):
        data_small.append(data[data["X_ray_image_name"] == filename])
        new_label = data[data["X_ray_image_name"] == filename]
        if(new_label["Label"].iloc[0] == "Pnemonia"):
            new_label = new_label["Label_1_Virus_category"].iloc[0]
        else:
            new_label = "Normal"
            
        label.append(new_label)
        images.append(img_arr)
        
    im.close()

images = np.array(images)

# %% shuffle the data
    
idx = np.arange(len(images))
np.random.shuffle(idx)

images = np.array(images)
images = images[idx]
label = np.array(label)
label = label[idx]


# %% create datasets

test_images = []
train_images = []
test_labels = []
train_labels = []

for i in range(len(images)):
    
    
    if(i < len(images) * 0.2):
        test_images.append(images[i])
        test_labels.append(label[i])
    else:
        train_images.append(images[i])
        train_labels.append(label[i])
        
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
test_images = np.array(test_images)

# %% reshpae images and rescale pixels

train_images = train_images.reshape((len(train_images), 150, 150, 3))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((len(test_images), 150, 150, 3))
test_images = test_images.astype('float32') / 255

# %% adapt labels to one hot encoding

train_labels[train_labels == "Virus"] = 0
train_labels[train_labels == "Normal"] = 1
train_labels[train_labels == "bacteria"] = 2 
train_labels = to_categorical(train_labels)
test_labels[test_labels == "Virus"] = 0
test_labels[test_labels == "Normal"] = 1
test_labels[test_labels == "bacteria"] = 2
test_labels = to_categorical(test_labels)

# %% create CNN 

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3), padding="same"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])


# %% train model

history = model.fit(train_images, train_labels, batch_size=64,
                    epochs=12, validation_split=0.2)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)


# %% print loss and accuracy plots

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(val_loss) + 1)

plt.figure()
plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'k-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure()
plt.plot(epochs, train_acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'k-', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# %% change filename to test wanted images

filename = "person1320_virus_2277.jpeg"
im = image.load_img(filename, target_size=(150, 150))
img_arr = image.img_to_array(im)
new_label = 0

image_label = data[data["X_ray_image_name"] == filename]["Label"]

if(len(image_label) != 0):
    data_small.append(data[data["X_ray_image_name"] == filename])
    new_label = data[data["X_ray_image_name"] == filename]
    if(new_label["Label"].iloc[0] == "Pnemonia"):
        new_label = new_label["Label_1_Virus_category"].iloc[0]
    else:
        new_label = "Normal"

if(new_label == "Normal"):
    new_label = 1
elif(new_label == "bacteria"):
    new_label = 2
else:
    new_label = 0

img_list = []
img_list.append(img_arr)
img_list = np.array(img_list)
test_image = img_list.reshape((len(img_list), 150, 150, 3))
test_image = test_image.astype('float32') / 255

prediction = model.predict(test_image)   
prediction = prediction.tolist() 

print(prediction.index(max(prediction)))
print("Real label: " + str(new_label))
im.close()