import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

import h5py
#import urllib.request
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

suffixes = ['B', 'KB', 'MB', 'GB']

def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    return data['X'], data['y']

def balanced_subsample(y, s):
    """Return a balanced subsample of the population"""
    sample = []
    # For every label in the dataset
    for label in np.unique(y):
        # Get the index of all images with a specific label
        images = np.where(y==label)[0]
        # Draw a random sample from the images
        random_sample = np.random.choice(images, size=s, replace=False)
        # Add the random sample to our subsample list
        sample += random_sample.tolist()
    return sample


def humansize(nbytes):
    if nbytes == 0: return '0 B'
    i = 0
    while nbytes >= 1024:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])

def chunking_dot(big_matrix, small_matrix, chunk_size=100):
    # Make a copy if the array is not already contiguous
    small_matrix = np.ascontiguousarray(small_matrix)
    R = np.empty((big_matrix.shape[0], big_matrix.shape[1], big_matrix.shape[2], small_matrix.shape[1]))
    for i in range(0, R.shape[0], chunk_size):
        end = i + chunk_size
        R[i:end] = np.dot(big_matrix[i:end], small_matrix)
    return R

def rgb2gray(images):
    """Convert images from rbg to grayscale
    """
    grayed = np.dot(images, np.array([0.2989, 0.5870, 0.1140]))
    return np.expand_dims(np.dot(images, [0.2989, 0.5870, 0.1140]), axis=3)


def main():
 # urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", "data/train_32x32.mat")
  #urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", "data/test_32x32.mat")
  #urllib.request.urlretrieve("http://ufldl.stanford.edu/housenumbers/extra_32x32.mat", "data/extra_32x32.mat")


  X_train, y_train = load_data('../datasets/train_32x32.mat')
  X_test, y_test = load_data('../datasets/test_32x32.mat')
  X_extra, y_extra = load_data('../datasets/extra_32x32.mat')

  print("Training", X_train.shape, y_train.shape)
  print("Test", X_test.shape, y_test.shape)
  print('Extra', X_extra.shape, y_extra.shape)

  # Transpose the image arrays
  X_train, y_train = X_train.transpose((3,0,1,2)), y_train[:,0]
  X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]
  X_extra, y_extra = X_extra.transpose((3,0,1,2)), y_extra[:,0]

  print("Training", X_train.shape)
  print("Test", X_test.shape)
  print("Extra", X_extra.shape)
  print('')

  # Calculate the total number of images
  num_images = X_train.shape[0] + X_test.shape[0] + X_extra.shape[0]

  print("Total Number of Images", num_images)

  print(np.unique(y_train))

  y_train[y_train == 10] = 0
  y_test[y_test == 10] = 0
  y_extra[y_extra == 10] = 0

  # Pick 400 samples per class from the training samples
  train_samples = balanced_subsample(y_train, 400)
  # Pick 200 samples per class from the extra dataset
  extra_samples = balanced_subsample(y_extra, 200)

  X_val, y_val = np.copy(X_train[train_samples]), np.copy(y_train[train_samples])

  # Remove the samples to avoid duplicates
  X_train = np.delete(X_train, train_samples, axis=0)
  y_train = np.delete(y_train, train_samples, axis=0)

  X_val = np.concatenate([X_val, np.copy(X_extra[extra_samples])])
  y_val = np.concatenate([y_val, np.copy(y_extra[extra_samples])])

  # Remove the samples to avoid duplicates
  X_extra = np.delete(X_extra, extra_samples, axis=0)
  y_extra = np.delete(y_extra, extra_samples, axis=0)

  X_train = np.concatenate([X_train, X_extra])
  y_train = np.concatenate([y_train, y_extra])
  X_test, y_test = X_test, y_test

  print("Training", X_train.shape, y_train.shape)
  print("Test", X_test.shape, y_test.shape)
  print('Validation', X_val.shape, y_val.shape)

  # Assert that we did not remove or add any duplicates
  assert(num_images == X_train.shape[0] + X_test.shape[0] + X_val.shape[0])

  # Transform the images to greyscale
  train_greyscale = rgb2gray(X_train).astype(np.float32)
  test_greyscale = rgb2gray(X_test).astype(np.float32)
  valid_greyscale = rgb2gray(X_val).astype(np.float32)

  # Keep the size before convertion
  size_before = (X_train.nbytes, X_test.nbytes, X_val.nbytes)

  # Size after transformation
  size_after = (train_greyscale.nbytes, test_greyscale.nbytes, valid_greyscale.nbytes)

  print("Dimensions")
  print("Training set", X_train.shape, train_greyscale.shape)
  print("Test set", X_test.shape, test_greyscale.shape)
  print("Validation set", X_val.shape, valid_greyscale.shape)
  print('')

  print("Data Type")
  print("Training set", X_train.dtype, train_greyscale.dtype)
  print("Test set", X_test.dtype, test_greyscale.dtype)
  print("Validation set", X_val.dtype, valid_greyscale.dtype)
  print('')

  print("Dataset Size")
  print("Training set", humansize(size_before[0]), humansize(size_after[0]))
  print("Test set", humansize(size_before[1]), humansize(size_after[1]))
  print("Validation set", humansize(size_before[2]), humansize(size_after[2]))

  # Fit the OneHotEncoder
  enc = OneHotEncoder().fit(y_train.reshape(-1, 1))

  # Transform the label values to a one-hot-encoding scheme
  y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
  y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
  y_val = enc.transform(y_val.reshape(-1, 1)).toarray()

  print("Training set", y_train.shape)
  print("Test set", y_test.shape)
  print("Training set", y_val.shape)

  # Create file
  h5f = h5py.File('/h/285/scarlettguo/CSC2515/ConvNet/data/SVHN_single.h5', 'w')

  # Store the datasets
  h5f.create_dataset('X_train', data=X_train)
  h5f.create_dataset('y_train', data=y_train)
  h5f.create_dataset('X_test', data=X_test)
  h5f.create_dataset('y_test', data=y_test)
  h5f.create_dataset('X_val', data=X_val)
  h5f.create_dataset('y_val', data=y_val)

  # Close the file
  print(h5f)
  h5f.close()

  # Create file
  h5f = h5py.File('/h/285/scarlettguo/CSC2515/ConvNet/data/SVHN_single_grey.h5', 'w')

  # Store the datasets
  h5f.create_dataset('X_train', data=train_greyscale)
  h5f.create_dataset('y_train', data=y_train)
  h5f.create_dataset('X_test', data=test_greyscale)
  h5f.create_dataset('y_test', data=y_test)
  h5f.create_dataset('X_val', data=valid_greyscale)
  h5f.create_dataset('y_val', data=y_val)

  # Close the file
  h5f.close()

if __name__ == "__main__":
  main()

