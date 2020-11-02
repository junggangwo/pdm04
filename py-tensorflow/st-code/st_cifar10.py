import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

from keras.datasets import cifar10

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Sidebar
st.sidebar.header('CIFAR-10')

# Show a random number
if st.sidebar.checkbox('Show a random Train image from CIFAR-10'):
    idx = np.random.randint(0, trainX.shape[0])
    image = trainX[idx]
    st.sidebar.image(image, caption=class_names[trainy[idx].item()], width=192)

if st.sidebar.checkbox('Show a random Test image from CIFAR-10'):
    idx = np.random.randint(0, testX.shape[0])
    image = testX[idx]
    st.sidebar.image(image, caption=class_names[testy[idx].item()], width=192)

# Main 
st.title('DL using CNN2D')
st.header('Dataset: CIFAR-10')

#spending a few lines to describe our dataset
st.text("""Dataset of 50,000 32x32x3 training images,
        and 10,000 test images.""")

# Information of cifar-10 dataset
if st.checkbox('Show images sizes'):
    st.write(f'##### X Train Shape: {trainX.shape}') 
    st.write(f'##### X Test Shape: {testX.shape}')
    st.write(f'##### Y Train Shape: {trainy.shape}')
    st.write(f'##### Y Test Shape: {testy.shape}')

st.write('***')

# display one random image:
st.subheader('Inspecting dataset')
if st.checkbox('Show random image from the train set'):
    idx = np.random.randint(0, trainX.shape[0])
    image = trainX[idx]
    st.image(image, caption=class_names[trainy[idx].item()], width=96)

if st.checkbox('Show random image from the test set'):
    idx = np.random.randint(0, testX.shape[0])
    image = testX[idx]
    st.image(image, caption=class_names[testy[idx].item()], width=96)

st.write('***')

if st.checkbox('Show 10 different image from the train set'):
    num_10 = np.unique(trainy, return_index=True)[1]
    images = trainX[num_10]
    fig = plt.figure(figsize=(10,6))

    for i in range(len(images)):
        # define subplot
        plt.subplot(2,5,1 + i) #, sharey=False)
        # plot raw pixel data
        plt.imshow(images[i])
        plt.title(class_names[i])
        plt.xticks([])
        plt.yticks([])
    
    plt.suptitle("10 different images", fontsize=18)
    st.pyplot()

if st.checkbox('Show 10 different image from the test set'):
    num_10 = np.unique(testy, return_index=True)[1]
    images = testX[num_10]
    fig = plt.figure(figsize=(10,6))

    for i in range(len(images)):
        # define subplot
        plt.subplot(2,5,1 + i) #, sharey=False)
        # plot raw pixel data
        plt.imshow(images[i])
        plt.title(class_names[i])
        plt.xticks([])
        plt.yticks([])
    
    plt.suptitle("10 different images", fontsize=18)
    st.pyplot()
