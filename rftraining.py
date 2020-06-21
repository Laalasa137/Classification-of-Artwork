import cv2
import numpy as np
import os
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

#Uncomment corresponding lines to perform artist, genre and style classification tasks repectively
train_path='./processed_Data/Artist/dataartisttrain_old'
val_path='./processed_Data/Artist/dataartistval_old'
# train_path='./processed_Data/Genre/datagenretrain_old'
# val_path='./processed_Data/Genre/datagenreval_old'
# train_path='./processed_Data/Style/datastyletrain_old'
# val_path='./processed_Data/Style/datastyleval_old'


training_names = os.listdir(train_path)

image_paths = []
image_classes = []
class_id = 0

def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

def imglist(path):
	return [os.path.join(path, f) for f in os.listdir(path)]
print("startttttttttttttttttttttttt read")
for training_name in training_names:
	dir = os.path.join(train_path, training_name)
	class_path = imglist(dir)
	image_paths+=class_path
	image_classes+=[class_id]*len(class_path)
	class_id+=1
print("endddddddddddddddddddddddddd read")
des_list = []

sift = cv2.xfeatures2d.SIFT_create(256)
print("startttttttttttttttttttttttt sift")
print(len(image_paths))
for image_path in image_paths:
	im = cv2.imread(image_path)
	if im is not None:
		im = to_gray(im)
		kpts, desc = sift.detectAndCompute(im, None)
		des_list.append((image_path, desc))
print("end sifttttttttttttttttttttttttttt")
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
		
		


descriptors_float = descriptors.astype(float)

k = 100
voc, variance = kmeans(descriptors_float, k, 1)

im_features = np.zeros((len(image_paths), k), "float32")

for i in range(len(image_paths)):
	words, distance = vq(des_list[i][1], voc)
	for w in words:
		im_features[i][w] +=1

nbr_occurances = np.sum((im_features > 0)*1, axis=0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurances + 1)), 'float32')
print("startttttttttttttttttttttttt randomforest")

stdslr = StandardScaler().fit(im_features)
im_features = stdslr.transform(im_features)


clf  = RandomForestClassifier(n_estimators=100, random_state=70)
clf.fit(im_features, np.array(image_classes))

#gets store in pickle file
joblib.dump((clf, training_names, stdslr, k, voc), "genrebovw.pkl", compress=3)
#load the same pickle file in testing file

print('yayyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy done')

