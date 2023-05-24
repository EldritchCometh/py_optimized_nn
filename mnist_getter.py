import gzip
import pickle
import numpy as np
from sklearn.datasets import fetch_openml

# Fetch the MNIST dataset
mnist = fetch_openml('mnist_784', parser='auto')
X_combined = mnist.data.values.tolist()
y_combined = mnist.target.values.tolist()

# Convert the data in the features to floats and the targets to one-hot encode
X_combined = [[float(f)/255 for f in x] for x in X_combined]
y_combined = [[float(i == int(y)) for i in range(10)] for y in y_combined]

# Split the data up between training and testing data
X_train, y_train = np.array(X_combined[:60000]), np.array(y_combined[:60000])
X_test, y_test = np.array(X_combined[60000:]), np.array(y_combined[60000:])

# Zip each feature set together with its target value
training_dataset = list(zip(X_train, y_train))
testing_dataset = list(zip(X_test, y_test))

# Dump it to the pickle
data = (training_dataset, testing_dataset)
with gzip.open("mnist_data.pkl.gz", 'wb') as f:
    pickle.dump(data, f)
