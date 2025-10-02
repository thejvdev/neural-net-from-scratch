import subprocess
import idx2numpy
import numpy as np

# Extract MNIST .gz archives
subprocess.run('cp ../datasets/mnist/*.gz . && gunzip *.gz', shell=True)

# Load image and label data
data = idx2numpy.convert_from_file('train-images-idx3-ubyte').astype(np.float32)
labels = idx2numpy.convert_from_file('train-labels-idx1-ubyte').astype(np.int32)

# Normalize pixel values to range [-1, 1]
data = (data / 255.0) * 2.0 - 1.0

# Save as CSV files
np.savetxt('data.csv', data.reshape(data.shape[0], -1), delimiter=',', fmt='%.6f')
np.savetxt('labels.csv', labels, delimiter=',', fmt='%d')

# Remove extracted binary files
subprocess.run('rm *ubyte', shell=True)
