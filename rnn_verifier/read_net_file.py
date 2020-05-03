import numpy as np
from fppoly import *
from elina_abstract0 import *
from elina_manager import *

from ctypes import *
from ctypes.util import *

import re


# open network file



def product(it):
    product = 1
    for x in it:
        product *= x
    return product

def runRepl(arg, repl):
    for a in repl:
        arg = arg.replace(a+"=", "'"+a+"':")
    return eval("{"+arg+"}")

def extract_mean(text):
    mean = ''
    m = re.search('mean=\[(.+?)\]', text)

    if m:
        means = m.group(1)
    mean_str = means.split(',')
    num_means = len(mean_str)
    mean_array = np.zeros(num_means)
    for i in range(num_means):
        mean_array[i] = np.float64(mean_str[i])
    return mean_array


def extract_std(text):
    std = ''
    m = re.search('std=\[(.+?)\]', text)
    if m:
        stds = m.group(1)
    std_str = stds.split(',')
    num_std = len(std_str)
    std_array = np.zeros(num_std)
    for i in range(num_std):
        std_array[i] = np.float64(std_str[i])
    return std_array


def extract_timestep(text):
    timestep = ''
    m = re.search(': (.+?)', text)

    if m:
        timestep = m.group(1)
    return np.int(timestep)


def numel(x):
    return product([int(i) for i in x.shape])


def parseVec(net):
    return np.array(eval(net.readline()[:-1]))


def myConst(vec):
    return tf.constant(vec.tolist(), dtype=tf.float64)


def permutation(W, h, w, c):
    m = np.zeros((h * w * c, h * w * c))

    column = 0
    for i in range(h * w):
        for j in range(c):
            m[i + j * h * w, column] = 1
            column += 1

    return np.matmul(W, m)

'''
tf.InteractiveSession().as_default()
'''
def read_tensorflow_net(net_file, in_len, is_trained_with_pytorch):
    mean = 0.0
    std = 0.0
    net = open(net_file,'r')

    y = None
    z1 = None
    z2 = None
    last_layer = None
    h,w,c = None, None, None
    is_conv = False
    weight_matrix = []
    bias = []

    #sunbing
    is_vanilarnn = False
    timestep = 0

    while True:
        curr_line = net.readline()[:-1]
        if curr_line in ["ReLU", "Sigmoid", "Tanh", "Affine"]:
            print(curr_line)
            W = None
            if (last_layer in ["Conv2D", "ParSumComplete", "ParSumReLU"]) and is_trained_with_pytorch:
                W = myConst(permutation(parseVec(net), h, w, c).transpose())
            else:
                W = parseVec(net)
                #W = myConst(W.transpose())
            b = parseVec(net)
            #b = myConst(b)

            if(curr_line=="Affine"):
                #x = tf.nn.bias_add(tf.matmul(tf.reshape(x, [1, numel(x)]),W), b)
                weight_out = W
                bias_out = b
            elif(curr_line=="ReLU"):
                weight_matrix.append(W)
                bias.append(b)
                #x = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.reshape(x, [1, numel(x)]),W), b))

            elif(curr_line=="Sigmoid"):
                x = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(tf.reshape(x, [1, numel(x)]),W), b))
            else:
                x = tf.nn.tanh(tf.nn.bias_add(tf.matmul(tf.reshape(x, [1, numel(x)]),W), b))
        elif curr_line == "":
            break
        else:
            raise Exception("Unsupported Operation: ", curr_line)
        last_layer = curr_line

    return weight_matrix, weight_out, bias, bias_out

'''
in_len: number of pixcels
in_len = input_size * timestep

input_size: number of input neuron
hidden_size: number of hidden neuron
output_size: number of outptu neuron
W_hh = hidden_size * hidden_size
W_ip = input_size * hidden_size
W_op = hidden_size * output_size
bias_hh = hidden_size
bias_op = output_size

dim = in_len + hidden_size

output:

weight = dim * hidden_size
weight_out = dim * output_size
bias = hidden_size
bias_out = output_size

example: hidden_size = 2
         input_size = 3
         in_len = 9
         dim = 11
'''
def convert_net_file(W_ip, W_hh, W_op, bias_hh, bias_op, timestep, input_size, hidden_size, out_size, out_file, mean, std, activation):
    in_len = input_size * timestep
    dim = in_len + hidden_size

    if activation != "ReLU":
        return
    # flattern weight matrix for each timestep:
    weight = []

    weight_pad = np.zeros(shape = (hidden_size, 1))

    for i in range (0, timestep):

        weight_i = W_hh
        for j in range (0, dim - (i + 1) * input_size - hidden_size):
            weight_i = np.concatenate((weight_i, weight_pad), axis = 1)

        weight_i = np.concatenate((weight_i, W_ip), axis = 1)

        for j in range (dim - i * input_size, dim):
            weight_i = np.concatenate((weight_i, weight_pad), axis = 1)
        weight.append(weight_i)

    weight_pad = np.zeros(shape=(out_size, 1))
    weight_out = W_op

    test_out = np.array([1.0, 1.0])
    for i in range (0, dim - hidden_size):
        if out_size == 1:
            weight_out = np.append(weight_out, ([0.0]))
        else:
            weight_out = np.concatenate((weight_out, weight_pad), axis = 1)

    out = open(out_file, 'w')

    for i in range(0, timestep):
        out.write("ReLU\n")

        itemList = []
        for item in weight[i]:
            itemList.append(list(item))

        out.write("{}\n".format(itemList))

        tmp_bias = list(bias_hh)
        out.write("{}\n".format(tmp_bias))

    # output layer
    out.write("Affine\n")

    if out_size == 1:
        tmp_w_op = list(weight_out)
        out.write("{}\n".format(tmp_w_op))
    else:
        itemList = []
        for item in weight_out:
            itemList.append(list(item))

        out.write("{}\n".format(itemList))

    tmp_bias = list(bias_op)
    out.write("{}\n".format(tmp_bias))

    out.close()

    return

'''
W_ip = np.array([[1.0, 1.0],[1.0, -1.0]])
W_hh = np.array([[0.1, -0.1],[-0.1, 0.1]])
W_op = np.array([[1.0, 0.0],[-1.0, 1.0]])
bias_hh = np.array([0.0, 0.0])
bias_op = np.array([0.0, -2.0])
'''

def read_input_file(in_file):

    net = open(in_file, 'r')
    timestep = 0
    is_vanilarnn = False
    mean = 0
    std = 0
    while True:
        curr_line = net.readline()[:-1]

        if 'ReLU' in curr_line:
            W_ip = parseVec(net)
            W_hh = parseVec(net)
            bias_hh = parseVec(net)

        elif 'Affine' in curr_line:
            W_op = parseVec(net)
            bias_op = parseVec(net)

        elif 'Normalize' in curr_line:
            mean = extract_mean(curr_line)
            std = extract_std(curr_line)
        elif 'time_step' in curr_line:
            timestep = extract_timestep(curr_line)
        elif curr_line == "Vanilla":
            is_vanilarnn = True
        elif curr_line == "":
            break
    return W_ip, W_hh, W_op, bias_hh, bias_op, timestep, is_vanilarnn, mean, std


def reorganize_input(input, hidden, timestep, step_size):
    output = hidden
    for i in range (timestep, 0, -1):
        index = (i - 1) * step_size
        for j in range(0, step_size):
            output.append(input[index + j])


    return output