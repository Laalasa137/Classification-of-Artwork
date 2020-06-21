import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray


def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read training images')
    for fields in classes:   
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            #print(image)
            #image = image.astype('uint8')
            if image is not None:
              image = image.astype('uint8')
              image = to_gray(image)
              image = cv2.resize(image, (image_size, image_size))
              image = image.astype(np.float32)
              image = np.multiply(image, 1.0 / 255.0)
              images.append(image)
              label = index
              # label[index] = 1.0
              labels.append(label)
              flbase = os.path.basename(fl)
              img_names.append(flbase)
              cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls


class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, val_path, image_size, classes, test_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  train_images, train_labels, train_img_names, train_cls = load_train(train_path, image_size, classes)
  print("Trained now shuffle begins")
  train_images, train_labels, train_img_names, train_cls = shuffle(train_images, train_labels, train_img_names, train_cls)
  print("Trained now shuffle ends")
  print("Validation starts")
  validation_images, validation_labels, validation_img_names, validation_cls = load_train(val_path, image_size, classes)
  print("Validation ends")
  print("Validation shuffle starts")
  validation_images, validation_labels, validation_img_names, validation_cls = shuffle(validation_images, validation_labels, validation_img_names, validation_cls)    
  print("Validation shuffle ends")
  if isinstance(test_size, float):
    test_size = int(test_size * train_images.shape[0])

  test_images = train_images[:test_size]
  test_labels = train_labels[:test_size]
  test_img_names = train_img_names[:test_size]
  test_cls = train_cls[:test_size]

  train_images = train_images[test_size:]
  train_labels = train_labels[test_size:]
  train_img_names = train_img_names[test_size:]
  train_cls = train_cls[test_size:]

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)
  data_sets.test = DataSet(test_images, test_labels, test_img_names, test_cls)
  print("data processing doneeeee")
  return data_sets