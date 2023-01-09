
'''
Loading weights in the entire model in one go
'''
# packages
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import cv2
import numpy as np
import pandas as pd
import os

# paths
weights_path = "/home/sansingh/san_home/temporaries/DjangoProjects/ProjectVGG16/HomeApp/static/vgg16/vgg16_weights.h5"
images_path = "/home/sansingh/Downloads/flickr_sample_dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_images/"
class_path = "/home/sansingh/san_home/GitHub/pretrained_mdoels/VGG-16/vgg16_class.csv"

# vgg16 model
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=1000, activation="softmax"))

# load weights
print("Loading VGG-16 weights...")
model.load_weights(weights_path)
print("Successfully loaded VGG-16 weights")

# load images
files = os.listdir(images_path)
files = files[:10]
images_data = np.ndarray((len(files), 224, 224, 3))
for i in range(len(files)):
    print(">>> ", (i + 1))
    img = cv2.imread(images_path + files[i])
    print("Original image size: ", img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    print("Resized image size: ", img.shape)
    images_data[i] = img

# make predictions
predict = model.predict(images_data)
predict_idx = np.argmax(predict, axis=1)

# load labels
labels_df = pd.read_csv(class_path)
labels = list(labels_df['labels'])

# get predicted class
pred_labels = [labels[i] for i in predict_idx]
print(pred_labels)