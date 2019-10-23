import numpy as np
import os
import cv2
import random
import pickle

#loading picture data
DATADIR = "C:/Users/coty/PycharmProjects/training/PetImages/"

#0<=>dog, 1<=>cat
CATEGORIES = ["Dog", "Cat"]

#meaning pictures are of size IMG_SIZE x IMG_SIZE
IMG_SIZE = 10
training_data_gray = []
training_data_rgb = []

#creating the gray and rgb picture sets of size IMG_SIZE x IMG_SIZE
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array_gray = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array_gray = cv2.resize(img_array_gray, (IMG_SIZE, IMG_SIZE))
                training_data_gray.append([new_array_gray, class_num])

                img_array_rgb = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                # changing default color sequence blue/green/red into red/green/blue
                img_array_rgb = cv2.cvtColor(img_array_rgb, cv2.COLOR_BGR2RGB)
                new_array_rgb = cv2.resize(img_array_rgb, (IMG_SIZE, IMG_SIZE))
                training_data_rgb.append([new_array_rgb, class_num])

            #some (few) pictures can not be processed and will be left out
            except Exception as e:
                pass

create_training_data()

#shuffeling the data to avoid overfitting, since the first half of the sets will consist of only dog-pics
A = list(range(0, len(training_data_gray)))
random.shuffle(A)
for i in range(len(A)):
    a = A[i]
    training_data_gray[i] = training_data_gray[a]
    training_data_rgb[i] = training_data_rgb[a]


X_gray = []
X_rgb = []
y = []

#collecting the rgb features and the general labels (0<=>dog, 1<=>cat)
for features, label in training_data_gray:
    X_gray.append(features)
    y.append(label)

#collecting the rgb features
for features, label in training_data_rgb:
    X_rgb.append(features)

#converting features and lables into convenient arrays
X_gray = np.array(X_gray).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_rgb = np.array(X_rgb).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)

pickle_out = open("X_gray.pickle", "wb")
pickle.dump(X_gray, pickle_out)
pickle_out.close()

pickle_out = open("X_rgb.pickle", "wb")
pickle.dump(X_rgb, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

