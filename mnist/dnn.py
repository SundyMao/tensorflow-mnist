import model
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.variable_scope("dnn"):
    x = tf.placeholder(tf.float32, [None, model.INPUT_NODE])
    y,_ = model.dnn(x, REGULARIZER)
# backward

y_ = tf.placeholder(tf.float32, [None, model.OUTPUT_NODE])
global_step = tf.Variable(0, trainable=False)

ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels = tf.argmax(y_, 1))
cem = tf.reduce_mean(ce)
loss = cem + tf.add_n(tf.get_collection('losses'))

learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

ema = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY, global_step)
ema_op = ema.apply(tf.trainable_variables())
with tf.control_dependencies([train_step, ema_op]):
    train_op = tf.no_op(name='train')

saver = tf.train.Saver()
    
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(STEPS):
        xs, ys = mnist.train.next_batch(BATCH_SIZE)
        _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_:ys})
        if i % 1000 == 0:
            print("After %d training step(s), loss on training batch is %g." % (step, loss_value))

    print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
        
    path = saver.save(sess, os.path.join(os.path.dirname(__file__), 'data', 'dnn.ckpt'), write_meta_graph=False, write_state=False)
    print("Saved:", path)

