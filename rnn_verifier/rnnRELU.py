import numpy as np
from fppoly import *
from elina_abstract0 import *
from elina_manager import *
import csv
from ctypes import *
from ctypes.util import *
from utils import *

from read_net_file import *
import re
import sys

sys.path.insert(0, '/Users/bing.sun/workspace/elina/ELINA-master-python/rnn_verifier/')

libc = CDLL(find_library('c'))
cstdout = c_void_p.in_dll(libc, '__stdoutp')
# input layer
epsilon = 0.01

input_pixel = 784
output_size = 10
hidden_size = 64

in_file =  "/Users/bing.sun/workspace/elina/ELINA-master-python/rnn_verifier/data/net/rnnRELU_input.pyt"
net_file = "/Users/bing.sun/workspace/elina/ELINA-master-python/rnn_verifier/data/net/rnnRELU_generated.pyt"
test_file = "/Users/bing.sun/workspace/elina/ELINA-master-python/rnn_verifier/data/test/mnist_test.csv"
'''
input_pixel = 9
output_size = 3
hidden_size = 4

in_file =  "/Users/bing.sun/workspace/elina/ELINA-master-python/rnn_verifier/data/net/in_test.pyt"
net_file = "/Users/bing.sun/workspace/elina/ELINA-master-python/rnn_verifier/data/net/in_generated.pyt"
test_file = "/Users/bing.sun/workspace/elina/ELINA-master-python/rnn_verifier/data/test/test.csv"
'''

'''
# data conversion
W_ip = np.array([[1.0, 1.0],[1.0, -1.0]])
W_hh = np.array([[0.1, -0.1],[-0.1, 0.1]])
W_op = np.array([[1.0, 0.0],[-1.0, 1.0]])
bias_hh = np.array([0.0, 0.0])
bias_op = np.array([0.0, -2.0])
'''

W_ip, W_hh, W_op, bias_hh, bias_op, timestep, is_vanilarnn, mean, std = read_input_file(in_file)

step_size = int(input_pixel / timestep)
dimension = input_pixel + hidden_size

convert_net_file(W_ip, W_hh, W_op, bias_hh, bias_op, timestep, step_size, hidden_size, output_size, net_file, mean, std, "ReLU")


weight_matrix, weight_out, bias, bias_out = read_tensorflow_net(net_file, input_pixel, True)
# intermediate layer (hidden layer)

weights_ptr = []
bias_ptr = []
#weights_op_ptr
#bias_op_ptr
predecessor = []

for i in range (0, timestep):
    np.ascontiguousarray(weight_matrix[i], dtype=np.double)
    weights_tmp = (weight_matrix[i].__array_interface__['data'][0] + np.arange(weight_matrix[i].shape[0]) * weight_matrix[i].strides[0]).astype(np.uintp)
    weights_ptr.append(weights_tmp)

    bias_tmp = bias[i]
    np.ascontiguousarray(bias_tmp, dtype=np.double)
    bias_ptr.append(bias_tmp)

    predecessor_tmp = [0]
    predecessor_tmp = (c_size_t * len(predecessor_tmp))()
    predecessor_tmp[0] = i
    predecessor.append(predecessor_tmp)

dim =  (c_size_t * dimension)()
for i in range (0, dimension):
    dim[i] = i

dim = ctypes.cast(dim, ctypes.POINTER(ctypes.c_size_t))

# output layer
weights_op_tmp = weight_out
np.ascontiguousarray(weights_op_tmp, dtype = np.double)
weights_op = (weights_op_tmp.__array_interface__['data'][0] + np.arange(weights_op_tmp.shape[0]) * weights_op_tmp.strides[0]).astype(np.uintp)

bias_op = bias_out
np.ascontiguousarray(bias_op, dtype = np.double)

pre_op = [-1]
pre_op = (c_size_t * len(pre_op))(timestep)

# intput layer

# read from data file

input_w = np.identity(dimension)

lpp_n = np.identity(dimension, dtype = np.double)
lpp_new = np.ascontiguousarray(lpp_n, dtype = np.double)


upp_n = np.identity(dimension, dtype = np.double)
upp_new = np.ascontiguousarray(upp_n, dtype = np.double)

lexpr_cst = np.array([0.0] * dimension)
np.ascontiguousarray(lexpr_cst, dtype = np.double)

uexpr_cst = np.array([0.0] * dimension)
np.ascontiguousarray(uexpr_cst, dtype = np.double)

expr_dims = []

for i in range (0, dimension):
    for j in range (0, dimension):
        expr_dims.append(j)

lexpr_size = np.array([dimension] * dimension)
uexpr_size = np.array([dimension] * dimension)

lexpr_dim = np.ascontiguousarray(expr_dims, dtype=np.uint64)
uexpr_dim = np.ascontiguousarray(expr_dims, dtype=np.uint64)

lexpr_size = lexpr_size.astype(np.uintp)
np.ascontiguousarray(lexpr_size, dtype=np.uintp)

uexpr_size = uexpr_size.astype(np.uintp)
np.ascontiguousarray(uexpr_size, dtype=np.uintp)

# for class comparision
compare_size = int((output_size) * (output_size - 1))
bias_l = np.array([0.0] * compare_size)
np.ascontiguousarray(bias_l, dtype=np.double)

pre_l = [0]
pre_l = (c_size_t * len(pre_l))(timestep + 1)


def get_compare_matrix(y1, y2, dim):
    matrix = np.array([0.0] * dim)
    for i in range(0, dim):
        if i == y1:
            matrix[i] = 1.0
        elif i == y2:
            matrix[i] = -1.0
    return matrix


weights_l = np.array([[0.0] * (dimension)] * (compare_size))

loop = 0
for label in range(0, output_size):
    for idx in range(0, output_size):
        if label != idx:
            weights_l[loop] = get_compare_matrix(label, idx, dimension)
            loop = loop + 1

np.ascontiguousarray(weights_l, dtype = np.double)
weights_l_ptr = (weights_l.__array_interface__['data'][0] + np.arange(weights_l.shape[0]) * weights_l.strides[0]).astype(np.uintp)
# input bounds

# read data file
csvfile = open(test_file, 'r')
tests = csv.reader(csvfile, delimiter=',')

total_image = 0
total_test_image = 0
total_verified_image = 0
for i, test in enumerate(tests):
    total_image = total_image + 1
    real_class = -1
    image = np.float64(test[1:len(test)]) / np.float64(255)

    hidden = [0.0] * hidden_size

    image = reorganize_input(image, hidden, timestep, step_size)

    inf = np.copy(image)
    sup = np.copy(image)

    #print(inf.tolist())


    normalize(inf, mean, std)
    normalize(sup, mean, std)
    '''

    inf = np.array([2.0] * dimension)
    sup = np.array([2.0] * dimension)
    '''

    man = fppoly_manager_alloc()
    element = fppoly_from_network_input_poly(man , 0, dimension, inf, sup, lpp_new, lexpr_cst, lexpr_dim, upp_new, uexpr_cst, uexpr_dim , dimension)

    rnn_handle_first_relu_layer(man, element, weights_ptr[0], bias_ptr[0], dim, hidden_size, dimension, predecessor[0])


    for k in range (1, timestep):
        rnn_handle_intermediate_relu_layer(man, element, weights_ptr[k], bias_ptr[k], dim, hidden_size, dimension, predecessor[k], True)

    rnn_handle_last_relu_layer(man, element, weights_op, bias_op, dim, output_size, dimension, pre_op, False, True)

    # verify property
    rnn_handle_last_relu_layer(man, element, weights_l_ptr, bias_l, dim, compare_size, dimension, pre_l, False, True)

    cmp_idx = 0
    for out_i in range(0, output_size):
        flag = True
        label = out_i
        for j in range(0, output_size):
            '''
            if label != j and not rnn_is_greater(man, element, label, j, (dimension), True):
                flag = False
                break
            '''
            if label != j:
                result = lb_for_neuron(man, element, (timestep + 1), (cmp_idx))
                cmp_idx = cmp_idx + 1
                if result < 0.0:
                    flag = False
                    #break
        if flag:
            real_class = label
            #break

    #elina_abstract0_fprint(cstdout, man, element, None)
    elina_abstract0_free(man, element)

    if real_class != int(test[0]):
        print("data image: {}, skipped and dominant class: {})".format((i + 1), real_class))
        continue

    total_test_image = total_test_image + 1
    print("data image: {}, real class: {}".format((i + 1), real_class))

    # pertubation
    dominant_class = -1
    inf = np.clip(np.array(image) - epsilon, 0, 1)
    sup = np.clip(np.array(image) + epsilon, 0, 1)

    normalize(inf, mean, std)
    normalize(sup, mean, std)

    man = fppoly_manager_alloc()
    element = fppoly_from_network_input_poly(man, 0, dimension, inf, sup, lpp_new, lexpr_cst, lexpr_dim, upp_new,
                                             uexpr_cst, uexpr_dim, dimension)

    rnn_handle_first_relu_layer(man, element, weights_ptr[0], bias_ptr[0], dim, hidden_size, dimension, predecessor[0])

    for k in range(1, timestep):
        rnn_handle_intermediate_relu_layer(man, element, weights_ptr[k], bias_ptr[k], dim, hidden_size, dimension,
                                           predecessor[k], True)

    rnn_handle_last_relu_layer(man, element, weights_op, bias_op, dim, output_size, dimension, pre_op, False, True)
    #elina_abstract0_fprint(cstdout, man, element, None)
    # verify property
    rnn_handle_last_relu_layer(man, element, weights_l_ptr, bias_l, dim, compare_size, dimension, pre_l, False, True)
    #elina_abstract0_fprint(cstdout, man, element, None)
    cmp_idx = 0
    for out_i in range(0, output_size):
        flag = True
        label = out_i
        for j in range(0, output_size):
            '''
            if label != j and not rnn_is_greater(man, element, label, j, (dimension), True):
                flag = False
                break
            '''
            if label != j:
                result = lb_for_neuron(man, element, (timestep + 1), (cmp_idx))
                cmp_idx = cmp_idx + 1
                if result < 0.0:
                    flag = False
                    #break
        if flag:
            dominant_class = label
            #break

    #elina_abstract0_fprint(cstdout, man, element, None)
    print("data image: {}, dominant class after perturbation: {}".format((i + 1), dominant_class))
    elina_abstract0_free(man, element)

    if dominant_class == int(test[0]):
        total_verified_image = total_verified_image + 1

total_verified_percentage = total_verified_image / total_test_image

print("Total input image: {}".format(total_image))
print("Total test image: {}".format(total_test_image))
print("Total verified image: {}".format(total_verified_image))
print("Total verified percentage: {}".format(total_verified_percentage))