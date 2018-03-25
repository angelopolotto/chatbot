# things we need for Tensorflow
import numpy as np
import tensorflow as tf
from tensorflow.python.training import saver

'''
Sources
base examples:
    https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/neural_network_raw.py
nn perceptron with epoch
    https://github.com/floydhub/tensorflow-examples/blob/master/3_NeuralNetworks/multilayer_perceptron.py
nminst implementations
    https://github.com/tensorflow/tensorflow/blob/7c36309c37b04843030664cdc64aca2bb7d6ecaa/tensorflow/contrib/learn/python/learn/datasets/mnist.py#L160
batch size
    https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network/164875
    https://stackoverflow.com/questions/46654424/how-to-calculate-optimal-batch-size
train dataset
    https://www.quora.com/What-is-a-training-data-set-test-data-set-in-machine-learning-What-are-the-rules-for-selecting-them
https://www.tensorflow.org/programmers_guide/saved_model
https://developers.google.com/machine-learning/glossary/#b
'''

training = np.load('dataset')

# create train and test lists
train_x = list(training[:, 0])
train_y = list(training[:, 1])
test_x = list(training[:, 0])
test_y = list(training[:, 1])

# Parameters
learning_rate = 0.1
batch_size = 8
training_epochs = 15
display_step = 100

# Network Parameters
num_input = len(train_x[0])  # train data input (0 or 1 vector (one hot) for each word found in the dataset)
n_hidden_1 = 8  # 1st layer number of neurons
n_hidden_2 = 8  # 2nd layer number of neurons
num_classes = len(train_y[0])  # total classes (0 or 1 vector (one hot) for each intent found in the )

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Construct model
chatbot_nn = neural_net(X)
prediction = tf.nn.softmax(chatbot_nn)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=chatbot_nn, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
def next_batch(X, Y, step, batch_size):
    x_len = len(X)
    index = (step-1) * batch_size
    if index + batch_size < x_len:
        batch_x = X[index:batch_size]
    else:
        batch_x = X[index:x_len]

    y_len = len(Y)
    index = (step - 1) * batch_size
    if index + batch_size < y_len:
        batch_y = X[index:batch_size]
    else:
        batch_y = X[index:y_len]

    return batch_x, batch_y


with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    # Training cycle

    for epoch in range(training_epochs):
        avg_cost = 0.
        num_steps = int(np.math.ceil(len(train_x) / batch_size))  #total batch
        for step in range(1, num_steps + 1):
            batch_x, batch_y = next_batch(X, Y, step, batch_size)
            # Run optimization op (backprop)
            _, c = sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            # Compute average loss
            avg_cost += c / step
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                  "{:.9f}".format(avg_cost))

    print("Optimization Finished!")
    # Save the variables to disk.
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)

    # # Test model
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={X: mnist.test.images,
                                        Y: mnist.test.labels}))]
