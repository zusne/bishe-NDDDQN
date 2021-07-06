import sys
print('-------------------model.py sys.path--------------------------------------------')
print(sys.path)
import tensorflow as tf
import numpy as np

# create convolutional networks
def create_conv_network(input_frames, trainable):
#input_frame:null,80,100,4
#conv1:null,20,25,16
#conv2:null,10,13,32
#conv3:null,10,13,32
    conv1_W = tf.get_variable(shape=[8, 12, 4, 16], name='conv1_W',
        trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
    conv1_b = tf.get_variable(shape=[16], name='conv1_b', trainable=trainable, initializer=tf.constant_initializer(
        0.05))
    conv1 = tf.nn.conv2d(input_frames, conv1_W, strides=[1, 4, 4, 1],
        padding='SAME', name='conv1')
    output1 = tf.nn.relu(conv1 + conv1_b, name='output1')

    conv2_W = tf.get_variable(shape=[4, 4, 16, 32], name='conv2_W',
        trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
    conv2_b = tf.get_variable(shape=[32], name='conv2_b', trainable=trainable, initializer=tf.constant_initializer(
        0.05))
    conv2 = tf.nn.conv2d(output1, conv2_W, strides=[1, 2, 2, 1],
        padding='SAME', name='conv2')
    output2 = tf.nn.relu(conv2 + conv2_b, name='output2')

    conv3_W = tf.get_variable(shape=[3, 3, 32, 32], name='conv3_W',
        trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
    conv3_b = tf.get_variable(shape=[32], name='conv3_b', trainable=trainable, initializer=tf.constant_initializer(
        0.05))
    conv3 = tf.nn.conv2d(output2, conv3_W, strides=[1, 1, 1, 1],
        padding='SAME', name='conv3')
    output3 = tf.nn.relu(conv3 + conv3_b, name='output3')

    flatten_shape = output3.get_shape()
    flat_output3_size = flatten_shape[1] * flatten_shape[2] * flatten_shape[3]
    flat_output3 = tf.reshape(output3, [-1, flat_output3_size], name='flat_output3')

    return flat_output3, flat_output3_size, [conv1_W, conv1_b, conv2_W, conv2_b, conv3_W, conv3_b]

# branching noisy dueling architecture
def create_duel_q_network(input_frames, num_actions, trainable):

    with tf.variable_scope('common'):
        flat_output, flat_output_size, parameter_list = create_conv_network(input_frames, trainable)

    ouputA, parameter_list_outputA = noisy_dense(flat_output, name='fcA',
                                                 input_size=4160, output_size=512, trainable=trainable,
                                                 activation_fn=tf.nn.relu)
    outputA2, parameter_list_outputA2 = noisy_dense(ouputA, name='fcA2',
                                                    input_size=512, output_size=num_actions,
                                                    trainable=trainable)
    ouputA_2, parameter_list_outputA_2 = noisy_dense(flat_output, name='fcA_2',
                                                     input_size=4160, output_size=512, trainable=trainable,
                                                     activation_fn=tf.nn.relu)
    outputA2_2, parameter_list_outputA2_2 = noisy_dense(ouputA_2, name='fcA2_2',
                                                        input_size=512, output_size=num_actions,
                                                        trainable=trainable)

    parameter_list += parameter_list_outputA + parameter_list_outputA2 + \
                      parameter_list_outputA_2 + parameter_list_outputA2_2 

    q_network = tf.nn.relu(outputA2, name='q_network')
    q_network_2 = tf.nn.relu(outputA2_2, name='q_network_2')
    return q_network, q_network_2, parameter_list

def create_model(window, input_shape, num_actions, model_name, create_network_fn, trainable):

    with tf.variable_scope(model_name, reuse=tf.AUTO_REUSE):
        input_frames = tf.placeholder(tf.float32, [None, input_shape[0],
                        input_shape[1], window], name ='input_frames')

        q_network, q_network2, parameter_list = create_network_fn(
            input_frames, num_actions, trainable)

        mean_max_Q1 = tf.reduce_mean(tf.reduce_max(q_network, axis=[1]), name='mean_max_Q1')
        mean_max_Q2 = tf.reduce_mean(tf.reduce_max(q_network2, axis=[1]), name='mean_max_Q2')

        action1 = tf.argmax(q_network, axis=1)
        action2 = tf.argmax(q_network2, axis=1)

        model = {
            'q_values1': q_network,
            'q_values2': q_network2,
            'input_frames': input_frames,
            'mean_max_Q1': mean_max_Q1,
            'mean_max_Q2': mean_max_Q2,
            'action1': action1,
            'action2': action2
        }
    return model, parameter_list

def noisy_dense(x, input_size, output_size, name, trainable, activation_fn=tf.identity):

    def f(x):
        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
    # Initialize \mu and \sigma
    mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(input_size, 0.5),
                                                maxval=1*1/np.power(input_size, 0.5))
    sigma_init = tf.constant_initializer(0.4/np.power(input_size, 0.5))

    p = tf.random_normal([input_size, 1])
    q = tf.random_normal([1, output_size])
    f_p = f(p); f_q = f(q)
    w_epsilon = f_p*f_q; b_epsilon = tf.squeeze(f_q)

    w_mu = tf.get_variable(name + "/w_mu", [input_size, output_size],
            initializer=mu_init, trainable=trainable)
    w_sigma = tf.get_variable(name + "/w_sigma", [input_size, output_size],
            initializer=sigma_init, trainable=trainable)
    w = w_mu + tf.multiply(w_sigma, w_epsilon)
    ret = tf.matmul(x, w)

    b_mu = tf.get_variable(name + "/b_mu", [output_size],
            initializer=mu_init, trainable=trainable)
    b_sigma = tf.get_variable(name + "/b_sigma", [output_size],
            initializer=sigma_init, trainable=trainable)
    b = b_mu + tf.multiply(b_sigma, b_epsilon)
    return activation_fn(ret + b), [w_mu, w_sigma, b_mu, b_sigma]
