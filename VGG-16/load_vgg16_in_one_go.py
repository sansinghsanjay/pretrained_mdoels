'''
Loading weights in the entire model in one go
'''
# packages
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import cv2
import numpy as np
import pandas as pd

# paths
weights_path = "C:/Users/Sanjay Singh/OneDrive/pretrained_model_weights/vgg16/vgg16_weights.h5"
cat_image_path = "C:/Users/Public/Documents/github_projects/pretrained_models/VGG-16/cat.jpg"
class_path = "C:/Users/Public/Documents/github_projects/pretrained_models/VGG-16/vgg16_class.csv"

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

# load image
img = cv2.imread(cat_image_path)
print("image shape: ", img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
print("image shape after reshape: ", img.shape)

# make prediction
print("Inferencing on the loaded image...")
p = model.predict(img)
p_class_id = np.argmax(p)

# load labels
labels_df = pd.read_csv(class_path)
labels = list(labels_df['labels'])
print("Predicted ID: ", p_class_id)
print("Label: ", labels[p_class_id])