from sklearn import datasets
from sklearn import neighbors
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from os import listdir
from sklearn.metrics import accuracy_score

data=datasets.load_iris()





def save_data(raw_folder):
    raw_folder = "dataset/trainingSet/"
    dest_size = (28,28)
    print("Bắt đầu xử lý ảnh...")

    pixels = []
    labels = []

    # Lặp qua các folder con trong thư mục raw
    for folder in listdir(raw_folder):
        if folder!='.DS_Store':
            print("Folder=",folder)
            # Lặp qua các file trong từng thư mục chứa các em
            for file in listdir(raw_folder  + folder):
                if file!='.DS_Store':
                    print("File=", file)
                    pixels.append(cv2.resize(cv2.imread(raw_folder+ folder +"/" + file,0),dsize=(28,28)))
                    labels.append(folder)

    pixels = np.array(pixels)
    labels = np.array(labels)


    return pixels , labels

x_train,y_train=save_data("dataset/trainingSet/")
x_test,y_test=save_data("dataset/testingSet/")


nsamples, nx, ny = x_train.shape
x_train = x_train.reshape((nsamples,nx*ny))
print(x_train.shape)

nsamples, nx, ny = x_test.shape
x_test = x_test.reshape((nsamples,nx*ny))

print(x_train.shape)

clf = neighbors.KNeighborsClassifier(n_neighbors = 20, p = 2)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print(y_pred[500:600])
print(y_test[500:600])
print((100*accuracy_score(y_test, y_pred)))

'''
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g=cv2.resize(gray, dsize=(28,28))
    nx, ny = g.shape
    g = g.reshape((1,nx*ny))
    clf = neighbors.KNeighborsClassifier(n_neighbors = 15, p = 2)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(g)
    print(y_pred)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
'''
