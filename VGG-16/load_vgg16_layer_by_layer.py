'''
Loading weights layer by layer
'''
# packages
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from PIL import Image
import numpy as np
import pandas as pd
import h5py

# paths
weights_path = "C:/Users/Sanjay Singh/OneDrive/pretrained_model_weights/vgg16/vgg16_weights.h5"
image_path = "C:/Users/Public/Documents/github_projects/pretrained_models/VGG-16/cat.jpg"
class_path = "C:/Users/Public/Documents/github_projects/pretrained_models/VGG-16/vgg16_class.csv"

# constants
IMG_W = 224
IMG_H = 224

# read image
print("Loading image...")
img = Image.open(image_path)
print("Successfully loaded image")

# resize image
img_resize = img.resize((IMG_W, IMG_H))
img_mat = np.array(img_resize)
img_mat = np.reshape(img_mat, (1, img_mat.shape[0], img_mat.shape[1], img_mat.shape[2]))

# vgg16 model
model = Sequential()
model.add(Conv2D(input_shape=(IMG_W,IMG_H,3),filters=64,kernel_size=(3,3),padding="same", activation="relu", trainable=False))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", trainable=False))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", trainable=False))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", trainable=False))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", trainable=False))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", trainable=False))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", trainable=False))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", trainable=False))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", trainable=False))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", trainable=False))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", trainable=False))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", trainable=False))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", trainable=False))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu", trainable=False))
model.add(Dense(units=4096,activation="relu", trainable=False))
model.add(Dense(units=1000, activation="softmax", trainable=False))

# read weights file
print("Reading VGG-16 weights...")
f_ptr = h5py.File(weights_path, "r")
print("Successfully read VGG-16 weights")

# load weights
print("Loading VGG-16 weights in the network...")
model.layers[0].set_weights([
	np.array(f_ptr['model_weights']['block1_conv1']['block1_conv1']['kernel:0']),
	np.array(f_ptr['model_weights']['block1_conv1']['block1_conv1']['bias:0'])
])
model.layers[1].set_weights([
	np.array(f_ptr['model_weights']['block1_conv2']['block1_conv2']['kernel:0']),
	np.array(f_ptr['model_weights']['block1_conv2']['block1_conv2']['bias:0'])
])
model.layers[3].set_weights([
	np.array(f_ptr['model_weights']['block2_conv1']['block2_conv1']['kernel:0']),
	np.array(f_ptr['model_weights']['block2_conv1']['block2_conv1']['bias:0'])
])
model.layers[4].set_weights([
	np.array(f_ptr['model_weights']['block2_conv2']['block2_conv2']['kernel:0']),
	np.array(f_ptr['model_weights']['block2_conv2']['block2_conv2']['bias:0'])
])
model.layers[6].set_weights([
	np.array(f_ptr['model_weights']['block3_conv1']['block3_conv1']['kernel:0']),
	np.array(f_ptr['model_weights']['block3_conv1']['block3_conv1']['bias:0'])
])
model.layers[7].set_weights([
	np.array(f_ptr['model_weights']['block3_conv2']['block3_conv2']['kernel:0']),
	np.array(f_ptr['model_weights']['block3_conv2']['block3_conv2']['bias:0'])
])
model.layers[8].set_weights([
	np.array(f_ptr['model_weights']['block3_conv3']['block3_conv3']['kernel:0']),
	np.array(f_ptr['model_weights']['block3_conv3']['block3_conv3']['bias:0'])
])
model.layers[10].set_weights([
	np.array(f_ptr['model_weights']['block4_conv1']['block4_conv1']['kernel:0']),
	np.array(f_ptr['model_weights']['block4_conv1']['block4_conv1']['bias:0'])
])
model.layers[11].set_weights([
	np.array(f_ptr['model_weights']['block4_conv2']['block4_conv2']['kernel:0']),
	np.array(f_ptr['model_weights']['block4_conv2']['block4_conv2']['bias:0'])
])
model.layers[12].set_weights([
	np.array(f_ptr['model_weights']['block4_conv3']['block4_conv3']['kernel:0']),
	np.array(f_ptr['model_weights']['block4_conv3']['block4_conv3']['bias:0'])
])
model.layers[14].set_weights([
	np.array(f_ptr['model_weights']['block5_conv1']['block5_conv1']['kernel:0']),
	np.array(f_ptr['model_weights']['block5_conv1']['block5_conv1']['bias:0'])
])
model.layers[15].set_weights([
	np.array(f_ptr['model_weights']['block5_conv2']['block5_conv2']['kernel:0']),
	np.array(f_ptr['model_weights']['block5_conv2']['block5_conv2']['bias:0'])
])
model.layers[16].set_weights([
	np.array(f_ptr['model_weights']['block5_conv3']['block5_conv3']['kernel:0']),
	np.array(f_ptr['model_weights']['block5_conv3']['block5_conv3']['bias:0'])
])
model.layers[19].set_weights([
	np.array(f_ptr['model_weights']['fc1']['fc1']['kernel:0']),
	np.array(f_ptr['model_weights']['fc1']['fc1']['bias:0'])
])
model.layers[20].set_weights([
	np.array(f_ptr['model_weights']['fc2']['fc2']['kernel:0']),
	np.array(f_ptr['model_weights']['fc2']['fc2']['bias:0'])
])
model.layers[21].set_weights([
	np.array(f_ptr['model_weights']['predictions']['predictions']['kernel:0']),
	np.array(f_ptr['model_weights']['predictions']['predictions']['bias:0'])
])
print("Successfully loaded VGG-16 weights")

# make prediction
print("Inferencing on the loaded image...")
p = model.predict(img_mat)
p_class_id = np.argmax(p)

# load labels
labels_df = pd.read_csv(class_path)
labels = list(labels_df['labels'])
print("Predicted ID: ", p_class_id)
print("Label: ", labels[p_class_id])