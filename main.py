import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from mnist import model

x = tf.placeholder("float", [None, 784])
sess = tf.Session()

# restore trained data
with tf.variable_scope("regression"):
    y1, variables = model.regression(x)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/regression.ckpt")

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convolutional.ckpt")

with tf.variable_scope("dnn"):
    y3,variables = model.dnn(x, None)
#    variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)
#    variables_to_restore = variable_averages.variables_to_restore()
# print variables
# print variables_to_restore
saver = tf.train.Saver(variables)
#saver = tf.train.Saver(variables_to_restore)
saver.restore(sess, "mnist/data/dnn.ckpt")

def regression(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()

def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()

def dnn(input):
    return sess.run(y3, feed_dict={x: input}).flatten().tolist()

# webapp
app = Flask(__name__)

@app.route('/api', methods=['POST'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output1 = regression(input)
    output2 = convolutional(input)
    output3 = dnn(input)
    return jsonify(results=[output1, output2, output3])

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)

