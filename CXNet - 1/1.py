import pandas as pd
import numpy as np
import os
import cv2
import pickle
import random

FILE_PATH = "C:/Users/Shad Humydee/Desktop/MSCSE/1. Summer 2020/CSE 6011 (Data Mining)/Submission/Project/Codebase_CDXL/_assets/Dataset - 2/metadata.csv"
IMAGES_PATH = "C:/Users/Shad Humydee/Desktop/MSCSE/1. Summer 2020/CSE 6011 (Data Mining)/Submission/Project/Codebase_CDXL/_assets/Dataset - 2//images"

df = pd.read_csv(FILE_PATH)
# print(df.shape)

#df_rows = len(df)


path = IMAGES_PATH

IMG_SIZE = 75

training_data = []


def create_training_data():
    i = 0
    #j = 0
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

        class_num = 0

        if df.iat[i, 4] == "COVID-19":
            class_num = 1
            #j += 1
            #print("Covid X-Ray at " + str(i))
        #print("Value of i = " + str(i))
        i += 1
        training_data.append([new_array, class_num])  # add this to our training_data
    #print("Value of j = " + str(j))

create_training_data()

#print(training_data[0])

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    #print("shad flag here!")
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(np.asarray(X), pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(np.asarray(y), pickle_out)
pickle_out.close()
