import sys

sys.path.insert(0, '/Users/bing.sun/workspace/elina/ELINA-master-python/rnn_verifier/')
sys.path.insert(0, '/Users/bing.sun/workspace/elina/ELINA-master-python/python_interface')

import numpy as np
from fppoly import *
from elina_abstract0 import *
from elina_manager import *
import csv
from ctypes import *
from ctypes.util import *
from utils import *

from read_net_file import *
from export_expr import *
import re
from lstm_tester import *

libc = CDLL(find_library('c'))
cstdout = c_void_p.in_dll(libc, '__stdoutp')

lstm_test_on = True
export_coef = True

# input layer
epsilon = 0.001

input_pixel = 784
output_size = 10
hidden_size = 16
hidden_gate = hidden_size * 4
step_size = 112

in_file =  "/Users/bing.sun/workspace/elina/ELINA-master-python/rnn_verifier/data/net/lstm_net_7_16.txt"
net_file = "/Users/bing.sun/workspace/elina/ELINA-master-python/rnn_verifier/data/net/rnnLSTM_generated_7_16.pyt"
test_file = "/Users/bing.sun/workspace/elina/ELINA-master-python/rnn_verifier/data/test/mnist_test.csv"

coeff_out_file = "/Users/bing.sun/workspace/elina/ELINA-master-python/rnn_verifier/data/test/coeff_out_lstm_{}_7_16.csv".format(epsilon)
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

convert_net_file(W_ip, W_hh, W_op, bias_hh, bias_op, timestep, step_size, hidden_size, output_size, net_file, mean, std, "LSTM")


weight_matrix, weight_out, bias, bias_out = read_tensorflow_net(net_file, input_pixel, True)
# intermediate layer (hidden layer)

weights_ptr = []
#bias_ptr
#weights_op_ptr
#bias_op_ptr
predecessor = []

for i in range (0, timestep):
    np.ascontiguousarray(weight_matrix[i], dtype=np.double)
    weights_tmp = (weight_matrix[i].__array_interface__['data'][0] + np.arange(weight_matrix[i].shape[0]) * weight_matrix[i].strides[0]).astype(np.uintp)
    weights_ptr.append(weights_tmp)

    predecessor_tmp = [0]
    predecessor_tmp = (c_size_t * len(predecessor_tmp))()
    predecessor_tmp[0] = i + 1
    predecessor.append(predecessor_tmp)

bias_tmp = bias[0]
#bias_tmp = np.array([10.0] * (hidden_size * 4))
bias_tmp = np.ascontiguousarray(bias_tmp, dtype=np.double)



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
pre_op = (c_size_t * len(pre_op))(timestep + 1)

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

# construct input layer

weights_ip = np.identity(dimension, dtype = np.double)
w_ip_tmp = np.ascontiguousarray(weights_ip, dtype = np.double)
weights_ip_ptr = (w_ip_tmp.__array_interface__['data'][0] + np.arange(w_ip_tmp.shape[0]) * w_ip_tmp.strides[0]).astype(np.uintp)

bias_ip = np.array([0.0] * dimension)
np.ascontiguousarray(bias_ip, dtype = np.double)


# for class comparision
compare_size = int((output_size) * (output_size - 1))
bias_l = np.array([0.0] * compare_size)
np.ascontiguousarray(bias_l, dtype=np.double)

pre_l = [0]
pre_l = (c_size_t * len(pre_l))(timestep + 2)


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

# out coefficient file
if export_coef == True:
    csvfile_coe = open(coeff_out_file, 'w')
    writer = csv.writer(csvfile_coe)

total_image = 0
total_test_image = 0
total_verified_image = 0

for i, test in enumerate(tests):
    total_image = total_image + 1
    real_class = -1
    image = np.float64(test[1:len(test)]) / np.float64(255)
    full_img = image

    hidden = [0.0] * hidden_size

    image = reorganize_input(image, hidden, timestep, step_size)

    #test
    #if lstm_test_on == True:
     #   image = np.array([1.0] * dimension)
     #   full_img = np.array([1.0] * input_pixel)


    inf = np.copy(image)
    sup = np.copy(image)


   # normalize(inf, mean, std)
   # normalize(sup, mean, std)

    pre_in = [1]
    pre_in = (c_size_t * len(pre_in))(-1)

    # test
    if lstm_test_on == True:



        tester_w = np.concatenate((W_hh, W_ip), axis=1)
        tester_b = bias_hh


        for i in range (0, timestep):
            start = i * step_size
            step_image = np.array([])
            j = start
            for j in range (start, start + step_size):
                step_image = np.append(step_image, full_img[j])


            if i == 0:
                hx1 = np.array([0.0] * hidden_size)
            else:
                hx1 = np.array(h1)

            hx1 = np.append(hx1, step_image)

            #print('timestep {}:'.format(i))
            if i == 0:
                ci1 = np.array([0.0] * hidden_size)
            else:
                ci1 = np.array(c1)
            c1, h1 = lstm_layer(hx1, ci1, tester_w, tester_b, hidden_size)

            #print('ct: {}\n'.format(c1))
            print('{} ht: {}\n'.format(i, h1))
            tester_wo = W_op
            tester_bo = bias_out

        tester_out = dense_op(h1, tester_wo, tester_bo, output_size)


    man = fppoly_manager_alloc()
    element = fppoly_from_network_input_poly(man , 0, dimension, inf, sup, lpp_new, lexpr_cst, lexpr_dim, upp_new, uexpr_cst, uexpr_dim , dimension)

    lstm_handle_first_layer_(man, element, weights_ip_ptr, bias_ip, dim, dimension, pre_in)

    for k in range (0, timestep):
        print(k)
        lstm_handle_intermediate_layer_(man, element, weights_ptr[k], bias_tmp, dim, dimension, hidden_size, predecessor[k], True)
        elina_abstract0_fprint(cstdout, man, element, None)

    lstm_handle_last_layer_(man, element, weights_op, bias_op, dim, hidden_size, output_size, pre_op, True)
    elina_abstract0_fprint(cstdout, man, element, None)

    if export_coef == True:
        lexpr_coeff, uexpr_coeff = get_expr_coeff(man, element, hidden_size, timestep, step_size, output_size, dimension, False)
        csvfile_coe.write('test image {}\n'.format(i + 1))
        csvfile_coe.write('lower bound expression coefficients\n')
        for w_idx in range (0, output_size):
            row = lexpr_coeff[w_idx]
            writer.writerow(row)
        csvfile_coe.write('upper bound expression coefficients\n')
        for w_idx in range (0, output_size):
            row = uexpr_coeff[w_idx]
            writer.writerow(row)

    # verify property
    lstm_handle_last_layer_(man, element, weights_l_ptr, bias_l, dim, dimension, compare_size, pre_l, True)

    #elina_abstract0_fprint(cstdout, man, element, None)


    cmp_idx = 0
    for out_i in range(0, output_size):
        flag = True
        label = out_i
        for j in range(0, output_size):

            if label != j:
                result = lb_for_neuron(man, element, (timestep + 2), (cmp_idx))
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

    #normalize(inf, mean, std)
    #normalize(sup, mean, std)

    man = fppoly_manager_alloc()
    element = fppoly_from_network_input_poly(man , 0, dimension, inf, sup, lpp_new, lexpr_cst, lexpr_dim, upp_new, uexpr_cst, uexpr_dim , dimension)

    lstm_handle_first_layer_(man, element, weights_ip_ptr, bias_ip, dim, dimension, pre_in)

    for k in range (0, timestep):
        lstm_handle_intermediate_layer_(man, element, weights_ptr[k], bias_tmp, dim, dimension, hidden_size, predecessor[k], True)

    lstm_handle_last_layer_(man, element, weights_op, bias_op, dim, dimension, output_size, pre_op, True)

    #elina_abstract0_fprint(cstdout, man, element, None)
    if export_coef == True:
        lexpr_coeff, uexpr_coeff = get_expr_coeff(man, element, hidden_size, timestep, step_size, output_size, dimension, False)

        csvfile_coe.write('pertubed: lower bound expression coefficients\n')
        for w_idx in range (0, output_size):
            row = lexpr_coeff[w_idx]
            writer.writerow(row)
        csvfile_coe.write('pertubed: upper bound expression coefficients\n')
        for w_idx in range (0, output_size):
            row = uexpr_coeff[w_idx]
            writer.writerow(row)

    # verify property
    lstm_handle_last_layer_(man, element, weights_l_ptr, bias_l, dim, dimension, compare_size, pre_l, True)

    elina_abstract0_fprint(cstdout, man, element, None)
    cmp_idx = 0
    for out_i in range(0, output_size):
        flag = True
        label = out_i
        for j in range(0, output_size):

            if label != j:
                result = lb_for_neuron(man, element, (timestep + 2), (cmp_idx))
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

    #if i == 5:
    #    break
if export_coef == True:
    csvfile_coe.close()
csvfile.close()

total_verified_percentage = total_verified_image / total_test_image

print("Perturbation setting: {}".format(epsilon))
print("Total input image: {}".format(total_image))
print("Total test image: {}".format(total_test_image))
print("Total verified image: {}".format(total_verified_image))
print("Total verified percentage: {}".format(total_verified_percentage))