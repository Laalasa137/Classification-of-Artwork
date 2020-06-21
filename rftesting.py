import cv2
import numpy as np
import os
import pylab as pl
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report

#Uncomment corresponding lines to perform artist, genre and style classification tasks repectively
#train_path='./processed_Data/Artist/dataartisttrain_old'
#test_path='./processed_Data/Artist/dataartistval_old'
#train_path='./processed_Data/Genre/datagenretrain_old'
#test_path='./processed_Data/Genre/datagenreval_old'
#train_path='./processed_Data/Style/datastyletrain_old'
#test_path='./processed_Data/Style/datastyleval_old'

testing_names = os.listdir(test_path)

#load the same pickle file from there
clf, classes_names, stdslr, k, voc = joblib.load("genrebovw.pkl")

image_paths = []
image_classes = []
class_id = 0

def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]
print("startttttttttttttttttttttttt read")
for testing_name in testing_names:
    dir = os.path.join(test_path, testing_name)
    class_path = imglist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1
print("enddddddddddddddddddddddddd read")
des_list = []

sift = cv2.xfeatures2d.SIFT_create(128)
print("startttttttttttttttttttttttt sift")
for image_path in image_paths:
    im = cv2.imread(image_path)
    if im is not None:
        im = to_gray(im)
        kpts, desc = sift.detectAndCompute(im, None)
        des_list.append((image_path, desc))
print("enddddddddddddddddddddddddd SIFT_create")
descriptors = des_list[0][1]


cnt = 1
f = 1
for image_path, descriptor in des_list[1:]:
    print(cnt)
    cnt = cnt+1
    #print(np.shape(descriptor))
    if descriptor is not None:
        descriptors = np.vstack((descriptors, descriptor))
    else:
        f = f+1
        


# for image_path, descriptor in des_list[1:]:
#     descriptors = np.vstack((descriptors, descriptor))

descriptors_float = descriptors.astype(float)
print("startttttttttttttttttttttttt kmeans")
k = 200
voc, variance = kmeans(descriptors_float, k, 1)

test_features = np.zeros((len(image_paths), k), "float32")

for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        test_features[i][w] +=1
print("endddddddddddddddddddddddddddd kmeansssssss")
nbr_occurances = np.sum((test_features > 0)*1, axis=0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurances + 1)), 'float32')
print("startttttttttttttttttttttttt predictions")
#using stdslr from pickled folder
test_features = stdslr.transform(test_features)

true_class = [classes_names[i] for i in image_classes]

#predict from the model
predictions = [classes_names[i] for i in clf.predict(test_features)]

# print(true_class)
# print(predictions)

# print("True Class = ", + str(true_class))
# print("Prediction = ", + str(predictions))
print("enddddddddddddddddddddddd predictions")
def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('confusion matrix')
    pl.colorbar()
    pl.show()

accuracy = accuracy_score(true_class, predictions)

print("Accuracy = ", accuracy)
cm  = confusion_matrix(true_class, predictions)

print(cm)

showconfusionmatrix(cm)

print(classification_report(true_class, predictions))




