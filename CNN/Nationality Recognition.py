import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import cv2
import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#Create Input Format
def read_asirra_subset(subset_dir, one_hot=True, sample_size=None):
    # 학습+검증 데이터셋을 읽어들임
    filename_list = os.listdir(subset_dir)
    set_size = len(filename_list)

    if sample_size is not None and sample_size < set_size:
        # sample_size가 명시된 경우, 원본 중 일부를 랜덤하게 샘플링함
        filename_list = np.random.choice(filename_list, size=sample_size, replace=False)
        set_size = sample_size
    else:
        # 단순히 filename list의 순서를 랜덤하게 섞음
        np.random.shuffle(filename_list)

    # 데이터 array들을 메모리 공간에 미리 할당함
    X_set = np.empty((set_size, 64, 64, 3), dtype=np.float32)    # (N, H, W, 3)
    y_set = np.empty((set_size), dtype=np.uint8)                   # (N,)

    for i, filename in enumerate(filename_list):
        if i % 1000 == 0:
            print('Reading subset data: {}/{}...'.format(i, set_size), end='\r')
        label = filename.split('.')[0]
        if label == 'cat':
            y = 0
        else:  # label == 'dog'
            y = 1
        file_path = os.path.join(subset_dir, filename)
        img = cv2.imread(file_path)    # shape: (H, W, 3), range: [0, 255]
        img = cv2.resize(img, (256, 256), mode='constant').astype(np.float32)    # (256, 256, 3), [0.0, 1.0]
        X_set[i] = img
        y_set[i] = y

    if one_hot:
        # 모든 레이블들을 one-hot 인코딩 벡터들로 변환함, shape: (N, num_classes)
        y_set_oh = np.zeros((set_size, 2), dtype=np.uint8)
        y_set_oh[np.arange(set_size), y_set] = 1
        y_set = y_set_oh
    print('\nDone')

    return X_set, y_set


# Collect Image Data
src_basic="./image/"
data=pd.read_csv("Valid_data.csv")
Nation=data["Nationality"].unique()

img=[]

for i in Nation:
    src=src_basic+i+"/*.png"
    img.extend(glob.glob(src))


# Create Model
model=Sequential()

# first conv and max pooling layers
model.add(Conv2D(16,(3,3),padding='same',input_shape=(64,64,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())

# second conv and max pooling layers
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())

# third conv and max pooling layers
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())

# forth conv and max pooling layers
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dense())