from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export
import gzip
import numpy as np
# read local gz files, convert to numpy function
def load_localData():
    
    path = '/Users/User/Desktop/Machine learning/handwritten_digits dataset'
    
    files = [
          'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']
    
    paths = []
    for fname in files:
        paths.append(get_file(fname, origin=None, cache_dir=path + fname, cache_subdir=path))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(\
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(\
    imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_localData()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)