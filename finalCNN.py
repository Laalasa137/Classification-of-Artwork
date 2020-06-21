import dataset
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import classification_report


###################################################################################
###################Uncomment this part to run capsulenetworks for artist classification
output_classes     = {0: 'Albrecht_Durer',
                     1: 'Boris_Kustodiev',
                     2: 'Camille_Pissarro',
                     3: 'Childe_Hassam',
                     4: 'Claude_Monet',
                     5: 'Edgar_Degas',
                     6: 'Eugene_Boudin',
                     7: 'Gustave_Dore',
                     8: 'Ilya_Repin',
                     9: 'Ivan_Aivazovsky',
                     10: 'Ivan_Shishkin',
                     11: 'John_Singer_Sargent',
                     12: 'Marc_Chagall',
                     13: 'Martiros_Saryan',
                     14: 'Nicholas_Roerich',
                     15: 'Pablo_Picasso',
                     16: 'Paul_Cezanne',
                     17: 'Pierre_Auguste_Renoir',
                     18: 'Pyotr_Konchalovsky',
                     19: 'Raphael_Kirchner',
                     20: 'Rembrandt',
                     21: 'Salvador_Dali',
                     22: 'Vincent_van_Gogh'}
train_path='./processed_Data/Artist/dataartisttrain_old'
val_path='./processed_Data/Artist/dataartistval_old'

###################################################################################
###################Uncomment this part to run capsulenetworks for genre classification                      
#output_classes     =  {0: 'abstract_painting',
#                        1: 'cityscape',
#                        2: 'genre_painting',
#                        3: 'illustration',
#                        4: 'landscape',
#                        5: 'nude_painting',
#                        6: 'portrait',
#                        7: 'religious_painting',
#                        8: 'sketch_and_study',
#                        9: 'still_life'}
# train_path='./processed_Data/Genre/datagenretrain_old'
# val_path='./processed_Data/Genre/datagenreval_old'
###################################################################################
###################Uncomment this part to run capsulenetworks for style classification
# output_classes = {0: 'Abstract_Expressionism',
#                   1: 'Action_painting',
#                   2: 'Analytical_Cubism',
#                   3: 'Art_Nouveau',
#                   4: 'Baroque',
#                   5: 'Color_Field_Painting',
#                   6: 'Contemporary_Realism',
#                   7: 'Cubism',
#                   8: 'Early_Renaissance',
#                   9: 'Expressionism',
#                   10: 'Fauvism',
#                   11: 'High_Renaissance',
#                   12: 'Impressionism',
#                   13: 'Mannerism_Late_Renaissance',
#                   14: 'Minimalism',
#                   15: 'Naive_Art_Primitivism',
#                   16: 'New_Realism',
#                   17: 'Northern_Renaissance',
#                   18: 'Pointillism',
#                   19: 'Pop_Art',
#                   20: 'Post_Impressionism',
#                   21: 'Realism',
#                   22: 'Rococo',
#                   23: 'Romanticism',
#                   24: 'Symbolism',
#                   25: 'Synthetic_Cubism',
#                   26: 'Ukiyo_e'}

# train_path='./processed_Data/Style/datastyletrain_old'
# val_path='./processed_Data/Style/datastyleval_old'
#######################################################################







start = time.time()
batch_size = 128
num_epochs = 80
iterations = 1 
#20% of the data will automatically be used for validation
test_size = 0.2
img_size = 100
num_channels = 3

img_width  = img_size
img_height = img_size
channels   = num_channels




if not os.path.exists(train_path):  
    print("No such directory")
    raise Exception
classes = os.listdir(train_path)
num_classes = len(classes)    
# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(train_path, val_path, img_size, classes, test_size)

# Display the stats
print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))
print("Number of files in Test-set:\t{}".format(len(data.test.labels)))


X_train = data.train.images
y_train = data.train.labels

X_test = data.test.images
y_test = data.test.labels

X_valid = data.valid.images
y_valid = data.valid.labels
# print(np.shape(X_train),np.shape(y_train))
# print(type(y_train[1]), y_train[:10])
def create_model():
    """
    Creates a simple sequential model
    """
    
    cnn = tf.keras.Sequential()
    
    cnn.add(tf.keras.layers.InputLayer(input_shape=(img_height,img_width,1)))
    
    # Normalization
    cnn.add(tf.keras.layers.BatchNormalization())
    
    # Conv + Maxpooling
    cnn.add(tf.keras.layers.Convolution2D(64, (4, 4), padding='same', activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Dropout
    cnn.add(tf.keras.layers.Dropout(0.1))
    
    # Conv + Maxpooling
    cnn.add(tf.keras.layers.Convolution2D(64, (4, 4), activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Dropout
    cnn.add(tf.keras.layers.Dropout(0.3))

    # Converting 3D feature to 1D feature Vektor
    cnn.add(tf.keras.layers.Flatten())

    # Fully Connected Layer
    cnn.add(tf.keras.layers.Dense(256, activation='relu'))

    # Dropout
    cnn.add(tf.keras.layers.Dropout(0.5))
    
    # Fully Connected Layer
    cnn.add(tf.keras.layers.Dense(64, activation='relu'))
    
    # Normalization
    cnn.add(tf.keras.layers.BatchNormalization())

    cnn.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    cnn.compile(loss='sparse_categorical_crossentropy', optimizer=tf.compat.v1.train.AdamOptimizer(), metrics=['accuracy'])

    return cnn

create_model().summary()



histories = []

for i in range(0,iterations):
    print('Running iteration: %i' % i)
    
    # Saving the best checkpoint for each iteration
    filepath = "Artist%i.hdf5" % i
    
    X_train_ = X_train
    X_val_ = X_valid
    y_train_ = y_train
    y_val_ = y_valid
    print(np.shape(X_train_))
    print(np.shape(y_train_))
    X_train = X_train_.reshape(X_train_.shape[0], 100, 100, 1)
    print(np.shape(X_train_))
    cnn = create_model()
    history = cnn.fit(
        X_train_, y_train_,
        batch_size=batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_data=(X_val_, y_val_),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
        ]
    )
    
    histories.append(history.history)


def get_avg(histories, his_key):
    tmp = []
    for history in histories:
        #print(type(history))
        #print(history)
        tmp.append(history[his_key][np.argmin(history['val_loss'])])
    return np.mean(tmp)
    
print('Training: \t%0.8f loss / %0.8f acc'   % (get_avg(histories,'loss'), get_avg(histories,'acc')))
print('Validation: \t%0.8f loss / %0.8f acc' % (get_avg(histories,'val_loss'), get_avg(histories,'val_acc')))

test_loss = []
test_accs = []

for i in range(0,iterations):
    cnn = create_model()
    cnn.load_weights("Artist%i.hdf5" % i)
    
    score = cnn.evaluate(X_test, y_test, verbose=0)
    test_loss.append(score[0])
    test_accs.append(score[1])
    
    print('Running final test with model %i: %0.4f loss / %0.4f acc' % (i,score[0],score[1]))
    
print('\nAverage loss / accuracy on testset: %0.4f loss / %0.5f acc' % (np.mean(test_loss),np.mean(test_accs)))
print('Standard deviation: (+-%0.4f) loss / (+-%0.4f) acc' % (np.std(test_loss),np.std(test_accs)))

RUN = 0 # you can choose one of the different models trained above
cnn = create_model()
cnn.load_weights("Artist%i.hdf5" % RUN)

def plot_train_val(title, history):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Accuracy
    ax1.set_title('Model accuracy - %s' % title)
    ax1.plot(history['acc'])
    ax1.plot(history['val_acc'])
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.legend(['train', 'validation'], loc='upper left')

    # Loss
    ax2.set_title('Model loss - %s' % title)
    ax2.plot(history['loss'])
    ax2.plot(history['val_loss'])
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.legend(['train', 'validation'], loc='upper left')

    fig.set_size_inches(20, 5)
    plt.show()

plot_train_val('Model %i' % RUN, histories[RUN])


def plot_confusion_matrix(cm, class_, title='Confusion matrix', cmap=plt.cm.Reds):
    """
    This function plots a confusion matrix
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(class_))
    plt.xticks(tick_marks, class_, rotation=90)
    plt.yticks(tick_marks, class_)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()

predictions = cnn.predict_classes(X_test, verbose=0)
plot_confusion_matrix(confusion_matrix(y_test, predictions), list(output_classes.keys()))

print(classification_report(y_test, predictions))