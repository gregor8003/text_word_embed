Neural network
--------------


The neural network has been implemented in Keras.

The network effectively has three layers:

* embedding layer, that extracts two embedding vectors, based on word indexes given as input to this layer

* "mean" lambda layer, that calculates element-wise mean of concatenated embedding vectors

* output layer, with ``sigmoid`` activation function

The ``adagrad`` is used as optimizer, and ``binary_crossentropy`` is used as
loss function.

The training data is presented once, during single epoch, and ``fit_generator``
is used to optimize learning time.

The network learns weights in embedding layer via backpropagation. These weights
constitute embedding vectors for each word.
