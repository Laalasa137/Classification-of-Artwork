import matplotlib
import matplotlib.pyplot as plt
import dataset
import numpy as np
import tensorflow as tf
import os
import time
tf.compat.v1.reset_default_graph()
np.random.seed(42)
tf.compat.v1.set_random_seed(42)

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
#Parameterssss
batch_size = 128
num_epochs = 80
iterations = 1 
#20% of the data will automatically be used for validation
test_size = 0.2
img_size = 28

start = time.time()
print('Starteddddd', start)

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

print('Doneeeee', time.time()-start)
X_train = data.train.images
y_train = data.train.labels

X_test = data.test.images
y_test = data.test.labels

X_valid = data.valid.images
y_valid = data.valid.labels

print('#training dataaaaa')
print(np.shape(X_train),np.shape(y_train))
print(type(y_train[1]), y_train[:10])


print('#testing dataaaaa')
print(np.shape(X_test),np.shape(y_test))
print(type(y_test[1]), y_test[:10])

print('#Validation dataaaaa')
print(np.shape(X_valid),np.shape(y_valid))
print(type(y_valid[1]), y_valid[:10])

n_samples = 5

print('And Corresponding Labelsssss')
print(y_train[:n_samples])

X = tf.compat.v1.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")


#primary capsuless

caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules
caps1_n_dims = 8


conv1_params = {
    "filters": 256,
    "kernel_size": 9,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
}

conv2_params = {
    "filters": caps1_n_maps * caps1_n_dims, # 256 convolutional filters
    "kernel_size": 9,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
}


conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)


caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims],
                       name="caps1_raw")

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keepdims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector

caps1_output = squash(caps1_raw, name="caps1_output")

#Digital Capsules Now

caps2_n_caps = 10
caps2_n_dims = 16

init_sigma = 0.1

W_init = tf.random.normal(
    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")


batch_size = tf.shape(X)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")


caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                             name="caps1_output_tiled")

caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                            name="caps2_predicted")


#Routing by agreement
#round 1
raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                       dtype=np.float32, name="raw_weights")
routing_weights = tf.nn.softmax(raw_weights, axis=2, name="routing_weights")


weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                   name="weighted_predictions")
weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True,
                             name="weighted_sum")                       

caps2_output_round_1 = squash(weighted_sum, axis=-2,
                              name="caps2_output_round_1")


#round2

caps2_output_round_1_tiled = tf.tile(
    caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
    name="caps2_output_round_1_tiled")

agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                      transpose_a=True, name="agreement")

raw_weights_round_2 = tf.add(raw_weights, agreement,
                             name="raw_weights_round_2")


routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                        axis=2,
                                        name="routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                           caps2_predicted,
                                           name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                     axis=1, keepdims=True,
                                     name="weighted_sum_round_2")
caps2_output_round_2 = squash(weighted_sum_round_2,
                              axis=-2,
                              name="caps2_output_round_2")

caps2_output = caps2_output_round_2

#Estimating class probabilities

def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keepdims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)


y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")

y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")


y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")



#Labels

y = tf.compat.v1.placeholder(shape=[None], dtype=tf.int64, name="y")

#computing margin loss


m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

T = tf.one_hot(y, depth=caps2_n_caps, name="T")

caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                              name="caps2_output_norm")

present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, 10),
                           name="present_error")


absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, 10),
                          name="absent_error")

L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")

margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

#Reconstruction

mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                               name="mask_with_labels")

reconstruction_targets = tf.cond(mask_with_labels, # condition
                                 lambda: y,        # if True
                                 lambda: y_pred,   # if False
                                 name="reconstruction_targets")


reconstruction_mask = tf.one_hot(reconstruction_targets,
                                 depth=caps2_n_caps,
                                 name="reconstruction_mask")

reconstruction_mask_reshaped = tf.reshape(
    reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1],
    name="reconstruction_mask_reshaped")


caps2_output_masked = tf.multiply(
    caps2_output, reconstruction_mask_reshaped,
    name="caps2_output_masked")

decoder_input = tf.reshape(caps2_output_masked,
                           [-1, caps2_n_caps * caps2_n_dims],
                           name="decoder_input")

#Decoder


n_hidden1 = 512
n_hidden2 = 1024
n_output = 28 * 28


with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                              activation=tf.nn.relu,
                              name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2,
                              activation=tf.nn.relu,
                              name="hidden2")
    decoder_output = tf.layers.dense(hidden2, n_output,
                                     activation=tf.nn.sigmoid,
                                     name="decoder_output")


X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output,
                               name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference,
                                    name="reconstruction_loss")

alpha = 0.05

loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

#Computing accuracies and initializing paramters
correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")


optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")

init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
valid_dataset =  tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))


#Traininggggggggggggggggg



def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

n_epochs = 80
batch_size = 100
restore_checkpoint = True

n_iterations_per_epoch =  np.shape(X_train)[0] // batch_size
n_iterations_validation = np.shape(X_valid)[0] // batch_size
best_loss_val = np.infty
checkpoint_path = "./my_capsule_network"


def plot_train_val_test(train_acc, val_acc, train_loss, val_loss):
   fig, (ax1, ax2) = plt.subplots(1, 2)

   # Accuracy
   ax1.set_title('Model accuracy')
   ax1.plot(train_acc)
   ax1.plot(val_acc)
   ax1.set_xlabel('epoch')
   ax1.set_ylabel('accuracy')
   ax1.legend(['train', 'validation'], loc='upper left')

   # Loss
   ax2.set_title('Model loss')
   ax2.plot(train_loss)
   ax2.plot(val_loss)
   ax2.set_xlabel('epoch')
   ax2.set_ylabel('loss')
   ax2.legend(['train', 'validation',], loc='upper left')

   fig.set_size_inches(20, 5)
   plt.show()
   







train_acc = []
val_acc = []
train_loss = []
val_loss = []



with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()

    for epoch in range(n_epochs):
        loss_trains = []
        acc_trains = []
        for iteration in range(1, n_iterations_per_epoch + 1):
            X_batch, y_batch = next_batch(batch_size, X_train, y_train)
            #print(train.shape)
            #X_batch, y_batch = train
            # Run the training operation and measure the loss:
            loss_train, acc_train = sess.run(
                [loss, accuracy],
                feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                           y: y_batch,
                           mask_with_labels: True})
            
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                      iteration, n_iterations_per_epoch,
                      iteration * 100 / n_iterations_per_epoch,
                      loss_train),
                  end="")
            loss_trains.append(loss_train)
            acc_trains.append(acc_train)
        loss_train = np.mean(loss_trains)
        acc_train = np.mean(acc_trains)
        # At the end of each epoch,
        # measure the validation loss and accuracy:
        loss_vals = []
        acc_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch, y_batch = next_batch(batch_size, X_valid, y_valid)
            #X_batch, y_batch = valid
            loss_val, acc_val = sess.run(
                    [loss, accuracy],
                    feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                               y: y_batch})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                      iteration, n_iterations_validation,
                      iteration * 100 / n_iterations_validation),
                  end=" " * 10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        
        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
            epoch + 1, acc_val * 100, loss_val,
            " (improved)" if loss_val < best_loss_val else ""))
        train_acc.append(acc_train)
        val_acc.append(acc_val)
        train_loss.append(loss_train)
        val_loss.append(loss_val)
        # And save the model if it improved:
        if loss_val < best_loss_val:
            save_path = saver.save(sess, checkpoint_path)
            best_loss_val = loss_val




n_iterations_test = np.shape(X_test)[0] // batch_size

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    loss_tests = []
    acc_tests = []
    for iteration in range(1, n_iterations_test + 1):
        X_batch, y_batch = next_batch(batch_size, X_test, y_test)
        #[X_batch, y_batch] = test
        loss_test, acc_test = sess.run(
                [loss, accuracy],
                feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                           y: y_batch})
        loss_tests.append(loss_test)
        acc_tests.append(acc_test)
        print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                  iteration, n_iterations_test,
                  iteration * 100 / n_iterations_test),
              end=" " * 10)
    loss_test = np.mean(loss_tests)
    acc_test = np.mean(acc_tests)
    print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
        acc_test * 100, loss_test))

#predictions = cnn.predict_classes(X_test, verbose=0)
#plot_confusion_matrix(confusion_matrix(y_test, predictions), list(output_classes.keys()))

#print(classification_report(y_test, predictions))

plot_train_val_test(train_acc, val_acc, train_loss, val_loss)