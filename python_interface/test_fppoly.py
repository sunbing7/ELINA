import numpy as np
from fppoly import *
from elina_abstract0 import *
from elina_manager import *

from ctypes import *
from ctypes.util import *

libc = CDLL(find_library('c'))
cstdout = c_void_p.in_dll(libc, '__stdoutp')


#debug

net_file = "/Users/bing.sun/workspace/elina/ELINA-master-python/rnn_verifier/data/net/rnnRELU_generated.pyt"
out_file = "/Users/bing.sun/workspace/elina/ELINA-master-python/rnn_verifier/data/net/rnnRELU_generated_c.pyt"
infile = open(net_file,'r')
outfile = open(out_file,'w')
while True:
    curr_line = infile.readline()[:-1]
    curr_line = curr_line.replace("[", "{")
    curr_line = curr_line.replace("]", "}")
    outfile.write(curr_line)
    outfile.write("\n")


    if curr_line == "":
        break
outfile.close()
# data rnn
dimension = 13
hidden_size = 4
output_size = 3
timestep = 3

inf = np.array([2.0] * dimension)
sup = np.array([2.0] * dimension)

lpp_new = np.identity(dimension, dtype=np.double)
lpp_new = np.ascontiguousarray(lpp_new, dtype=np.double)

upp_new = np.identity(dimension, dtype=np.double)
upp_new = np.ascontiguousarray(upp_new, dtype=np.double)

lexpr_cst = np.array([0.0] * dimension)
np.ascontiguousarray(lexpr_cst, dtype=np.double)

uexpr_cst = np.array([0.0] * dimension)
np.ascontiguousarray(uexpr_cst, dtype=np.double)

expr_dims = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
lexpr_dim = np.ascontiguousarray(expr_dims, dtype=np.uint64)
uexpr_dim = np.ascontiguousarray(expr_dims, dtype=np.uint64)

lexpr_size = np.array([13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13])
lexpr_size = lexpr_size.astype(np.uintp)
np.ascontiguousarray(lexpr_size, dtype=np.uintp)

uexpr_size = np.array([13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13])
uexpr_size = uexpr_size.astype(np.uintp)
np.ascontiguousarray(uexpr_size, dtype=np.uintp)


# layer 1
weights_1 = np.array([[0.1, 0.1, -0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.1, 0.1, -0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                      [-0.1, -0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0], [-0.1, -0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0]])
np.ascontiguousarray(weights_1, dtype = np.double)
weights_l1 = (weights_1.__array_interface__['data'][0] + np.arange(weights_1.shape[0]) * weights_1.strides[0]).astype(np.uintp)

bias_l1 = np.array([0.0, 0.0, 0.0, 0.0])
np.ascontiguousarray(bias_l1, dtype = np.double)

dim_l1 = (c_size_t * dimension)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
dim_l1 = ctypes.cast(dim_l1, ctypes.POINTER(ctypes.c_size_t))

pre_l1 = [-1]
pre_l1 = (c_size_t * len(pre_l1))(-1)


# layer 2
weights_2 = np.array([[0.1, 0.1, -0.1, -0.1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.1, 0.1, -0.1, -0.1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                      [-0.1, -0.1, 0.1, 0.1, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0], [-0.1, -0.1, 0.1, 0.1, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0]])
np.ascontiguousarray(weights_2, dtype = np.double)
weights_l2 = (weights_2.__array_interface__['data'][0] + np.arange(weights_2.shape[0]) * weights_2.strides[0]).astype(np.uintp)

bias_l2 = np.array([0.0, 0.0, 0.0, 0.0])
np.ascontiguousarray(bias_l2, dtype = np.double)

dim_l2 = (c_size_t * dimension)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
dim_l2 = ctypes.cast(dim_l1, ctypes.POINTER(ctypes.c_size_t))

pre_l2 = [1]
pre_l2 = (c_size_t * len(pre_l2))(1)


# layer 3
weights_3 = np.array([[0.1, 0.1, -0.1, -0.1, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.1, -0.1, -0.1, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [-0.1, -0.1, 0.1, 0.1, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.1, -0.1, 0.1, 0.1, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
np.ascontiguousarray(weights_3, dtype = np.double)
weights_l3 = (weights_3.__array_interface__['data'][0] + np.arange(weights_3.shape[0]) * weights_3.strides[0]).astype(np.uintp)

bias_l3 = np.array([0.0, 0.0, 0.0, 0.0])
np.ascontiguousarray(bias_l3, dtype = np.double)

dim_l3 = (c_size_t * dimension)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
dim_l3 = ctypes.cast(dim_l1, ctypes.POINTER(ctypes.c_size_t))

pre_l3 = [2]
pre_l3 = (c_size_t * len(pre_l3))(2)


# output layer
weights_out = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
np.ascontiguousarray(weights_out, dtype = np.double)
weights_op = (weights_out.__array_interface__['data'][0] + np.arange(weights_out.shape[0]) * weights_out.strides[0]).astype(np.uintp)

bias_op = np.array([0.0, 0.0, 0.0])
np.ascontiguousarray(bias_op, dtype = np.double)

dim_op = (c_size_t * dimension)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
dim_op = ctypes.cast(dim_op, ctypes.POINTER(ctypes.c_size_t))

pre_op = [3]
pre_op = (c_size_t * len(pre_op))(3)

# for class comparision
compare_size = int((output_size ) * (output_size - 1))
bias_l = np.array([0.0] * compare_size)
np.ascontiguousarray(bias_l, dtype = np.double)

dim_l = (c_size_t * dimension)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
dim_l = ctypes.cast(dim_l, ctypes.POINTER(ctypes.c_size_t))

pre_l = [4]
pre_l = (c_size_t * len(pre_l))(4)


def get_compare_matrix(y1, y2, dim):
    matrix = np.array([0.0] * dim)
    for i in range (0, dim):
        if i == y1:
            matrix[i] = 1.0
        elif i == y2:
            matrix[i] = -1.0
    return matrix

weights_i = np.array([[0.0] * (dimension)] * (compare_size))

loop = 0
for label in range (0, output_size):
    for idx in range (0, output_size):
        if label != idx:
            weights_i[loop] = get_compare_matrix(label, idx, dimension)
            loop = loop + 1
'''
weights_i = np.array([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
'''
np.ascontiguousarray(weights_i, dtype = np.double)
weights_i_ptr = (weights_i.__array_interface__['data'][0] + np.arange(weights_i.shape[0]) * weights_i.strides[0]).astype(np.uintp)


man = fppoly_manager_alloc()
element = fppoly_from_network_input_poly(man , 0, dimension, inf, sup, lpp_new, lexpr_cst, lexpr_dim, upp_new, uexpr_cst, uexpr_dim , 11)

rnn_handle_first_relu_layer(man, element, weights_l1, bias_l1, dim_l1, hidden_size, dimension, pre_l1)

rnn_handle_intermediate_relu_layer(man, element, weights_l2, bias_l2, dim_l2, hidden_size, dimension, pre_l2, True)

rnn_handle_intermediate_relu_layer(man, element, weights_l3, bias_l3, dim_l3, hidden_size, dimension, pre_l3, True)

rnn_handle_last_relu_layer(man, element, weights_op, bias_op, dim_op, 3, dimension, pre_op, False, True)
#rnn_handle_last_relu_layer(man, element, weights_op, bias_op, dim_op, 3, dimension, pre_l, False, True)
# compare output class

rnn_handle_last_relu_layer(man, element, weights_i_ptr, bias_l, dim_l, compare_size, dimension, pre_l, False, True)

lb_1 = lb_for_neuron(man, element, 4, 0)# 0,1
lb_2 = lb_for_neuron(man, element, 4, 1)# 0,2
lb_3 = lb_for_neuron(man, element, 4, 2)# 1,0
lb_4 = lb_for_neuron(man, element, 4, 3)# 1,2
lb_5 = lb_for_neuron(man, element, 4, 4)# 2,0
lb_6 = lb_for_neuron(man, element, 4, 5)# 2,1

cmp_idx = 0
for out_i in range(output_size):
    flag = True
    label = out_i
    for j in range(output_size):
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
                break
    if flag:
        dominant_class = label
        break
'''
result = rnn_is_greater(man, element, 0, 1, dimension, True)
result = rnn_is_greater(man, element, 0, 2, dimension, True)
result = rnn_is_greater(man, element, 1, 0, dimension, True)
result = rnn_is_greater(man, element, 1, 2, dimension, True)
result = rnn_is_greater(man, element, 2, 0, dimension, True)
result = rnn_is_greater(man, element, 2, 1, dimension, True)
'''
elina_abstract0_fprint(cstdout, man, element, None)


'''
# input layer

inf = np.array([2.0] * 11)
sup = np.array([2.0] * 11)

lpp_new = np.identity(11, dtype=np.double)
lpp_new = np.ascontiguousarray(lpp_new, dtype=np.double)

upp_new = np.identity(11, dtype=np.double)
upp_new = np.ascontiguousarray(upp_new, dtype=np.double)

lexpr_cst = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
np.ascontiguousarray(lexpr_cst, dtype=np.double)

uexpr_cst = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
np.ascontiguousarray(uexpr_cst, dtype=np.double)

expr_dims = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
lexpr_dim = np.ascontiguousarray(expr_dims, dtype=np.uint64)
uexpr_dim = np.ascontiguousarray(expr_dims, dtype=np.uint64)

lexpr_size = np.array([11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11])
lexpr_size = lexpr_size.astype(np.uintp)
np.ascontiguousarray(lexpr_size, dtype=np.uintp)

uexpr_size = np.array([11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11])
uexpr_size = uexpr_size.astype(np.uintp)
np.ascontiguousarray(uexpr_size, dtype=np.uintp)


# layer 1
weights_1 = np.array([[0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [-0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0]])
np.ascontiguousarray(weights_1, dtype = np.double)
weights_l1 = (weights_1.__array_interface__['data'][0] + np.arange(weights_1.shape[0]) * weights_1.strides[0]).astype(np.uintp)

bias_l1 = np.array([0.0, 0.0])
np.ascontiguousarray(bias_l1, dtype = np.double)

dim_l1 = (c_size_t * 11)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
dim_l1 = ctypes.cast(dim_l1, ctypes.POINTER(ctypes.c_size_t))

pre_l1 = [-1]
pre_l1 = (c_size_t * len(pre_l1))(-1)


# layer 2
weights_2 = np.array([[0.1, -0.1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [-0.1, 0.1, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0]])
np.ascontiguousarray(weights_2, dtype = np.double)
weights_l2 = (weights_2.__array_interface__['data'][0] + np.arange(weights_2.shape[0]) * weights_2.strides[0]).astype(np.uintp)

bias_l2 = np.array([0.0, 0.0])
np.ascontiguousarray(bias_l2, dtype = np.double)

dim_l2 = (c_size_t * 11)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
dim_l2 = ctypes.cast(dim_l1, ctypes.POINTER(ctypes.c_size_t))

pre_l2 = [1]
pre_l2 = (c_size_t * len(pre_l2))(1)


# layer 3
weights_3 = np.array([[0.1, -0.1, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.1, 0.1, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
np.ascontiguousarray(weights_3, dtype = np.double)
weights_l3 = (weights_3.__array_interface__['data'][0] + np.arange(weights_3.shape[0]) * weights_3.strides[0]).astype(np.uintp)

bias_l3 = np.array([0.0, 0.0])
np.ascontiguousarray(bias_l3, dtype = np.double)

dim_l3 = (c_size_t * 11)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
dim_l3 = ctypes.cast(dim_l1, ctypes.POINTER(ctypes.c_size_t))

pre_l3 = [2]
pre_l3 = (c_size_t * len(pre_l3))(2)


# output layer
weights_out = np.array([[-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
np.ascontiguousarray(weights_out, dtype = np.double)
weights_op = (weights_out.__array_interface__['data'][0] + np.arange(weights_out.shape[0]) * weights_out.strides[0]).astype(np.uintp)

bias_op = np.array([0.0, 0.0])
np.ascontiguousarray(bias_op, dtype = np.double)

dim_op = (c_size_t * 11)(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
dim_op = ctypes.cast(dim_op, ctypes.POINTER(ctypes.c_size_t))

pre_op = [3]
pre_op = (c_size_t * len(pre_op))(3)

man = fppoly_manager_alloc()
element = fppoly_from_network_input_poly(man , 0, 11, inf, sup, lpp_new, lexpr_cst, lexpr_dim, upp_new, uexpr_cst, uexpr_dim , 11)

rnn_handle_first_relu_layer(man, element, weights_l1, bias_l1, dim_l1, 2, 11, pre_l1)

rnn_handle_intermediate_relu_layer(man, element, weights_l2, bias_l2, dim_l2, 2, 11, pre_l2, True)

rnn_handle_intermediate_relu_layer(man, element, weights_l3, bias_l3, dim_l3, 2, 11, pre_l3, True)

rnn_handle_last_relu_layer(man, element, weights_op, bias_op, dim_op, 2, 11, pre_op, False, True)

result = rnn_is_greater(man, element, 1, 0, 11, True)


elina_abstract0_fprint(cstdout, man, element, None)
'''
'''
inf = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
sup = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

input_weights = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]])
np.ascontiguousarray(input_weights, dtype=np.double)

lpp = (input_weights.__array_interface__['data'][0] + np.arange(input_weights.shape[0]) * input_weights.strides[0]).astype(np.uintp)

lpp_new = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
lpp_new = np.ascontiguousarray(lpp_new, dtype=np.double)

upp = (input_weights.__array_interface__['data'][0] + np.arange(input_weights.shape[0]) * input_weights.strides[0]).astype(np.uintp)

upp_new = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
upp_new = np.ascontiguousarray(upp_new, dtype=np.double)

lexpr_cst = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
np.ascontiguousarray(lexpr_cst, dtype=np.double)

uexpr_cst = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
np.ascontiguousarray(uexpr_cst, dtype=np.double)

expr_dims = np.array([0, 1, 2, 3, 4, 5, 6, 7,
                      0, 1, 2, 3, 4, 5, 6, 7,
                      0, 1, 2, 3, 4, 5, 6, 7,
                      0, 1, 2, 3, 4, 5, 6, 7,
                      0, 1, 2, 3, 4, 5, 6, 7,
                      0, 1, 2, 3, 4, 5, 6, 7,
                      0, 1, 2, 3, 4, 5, 6, 7,
                      0, 1, 2, 3, 4, 5, 6, 7])
lexpr_dim = np.ascontiguousarray(expr_dims, dtype=np.uint64)
uexpr_dim = np.ascontiguousarray(expr_dims, dtype=np.uint64)

lexpr_size = np.array([8, 8, 8, 8, 8, 8, 8, 8])
lexpr_size = lexpr_size.astype(np.uintp)
np.ascontiguousarray(lexpr_size, dtype=np.uintp)

uexpr_size = np.array([8, 8, 8, 8, 8, 8, 8, 8])
uexpr_size = uexpr_size.astype(np.uintp)
np.ascontiguousarray(uexpr_size, dtype=np.uintp)


# layer 1
weights_1 = np.array([[0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                       [-0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]])
np.ascontiguousarray(weights_1, dtype = np.double)
weights_l1 = (weights_1.__array_interface__['data'][0] + np.arange(weights_1.shape[0]) * weights_1.strides[0]).astype(np.uintp)

bias_l1 = np.array([0.0, 0.0])
np.ascontiguousarray(bias_l1, dtype = np.double)

dim_l1 = (c_size_t * 8)(0, 1, 2, 3, 4, 5, 6, 7)
dim_l1 = ctypes.cast(dim_l1, ctypes.POINTER(ctypes.c_size_t))

pre_l1 = [-1]
pre_l1 = (c_size_t * len(pre_l1))(-1)


# layer 2
weights_2 = np.array([[0.1, -0.1, 0, 0, 1.0, 1.0, 0, 0],[-0.1, 0.1, 0, 0, 1.0, -1.0, 0, 0]])
np.ascontiguousarray(weights_2, dtype = np.double)
weights_l2 = (weights_2.__array_interface__['data'][0] + np.arange(weights_2.shape[0]) * weights_2.strides[0]).astype(np.uintp)

bias_l2 = np.array([0.0, 0.0])
np.ascontiguousarray(bias_l2, dtype = np.double)

dim_l2 = (c_size_t * 8)(0, 1, 2, 3, 4, 5, 6, 7)
dim_l2 = ctypes.cast(dim_l1, ctypes.POINTER(ctypes.c_size_t))

pre_l2 = [1]
pre_l2 = (c_size_t * len(pre_l2))(1)


# layer 3
weights_3 = np.array([[0.1, -0.1, 1.0, 1.0, 0, 0, 0, 0],[-0.1, 0.1, 1.0, -1.0, 0, 0, 0, 0]])
np.ascontiguousarray(weights_3, dtype = np.double)
weights_l3 = (weights_3.__array_interface__['data'][0] + np.arange(weights_3.shape[0]) * weights_3.strides[0]).astype(np.uintp)

bias_l3 = np.array([0.0, 0.0])
np.ascontiguousarray(bias_l3, dtype = np.double)

dim_l3 = (c_size_t * 8)(0, 1, 2, 3, 4, 5, 6, 7)
dim_l3 = ctypes.cast(dim_l1, ctypes.POINTER(ctypes.c_size_t))

pre_l3 = [2]
pre_l3 = (c_size_t * len(pre_l3))(2)


# output layer
weights_out = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
np.ascontiguousarray(weights_out, dtype = np.double)
weights_op = (weights_out.__array_interface__['data'][0] + np.arange(weights_out.shape[0]) * weights_out.strides[0]).astype(np.uintp)

bias_op = np.array([0.0, -2.0])
np.ascontiguousarray(bias_op, dtype = np.double)

dim_op = (c_size_t * 8)(0, 1, 2, 3, 4, 5, 6, 7)
dim_op = ctypes.cast(dim_op, ctypes.POINTER(ctypes.c_size_t))

pre_op = [3]
pre_op = (c_size_t * len(pre_op))(3)

man = fppoly_manager_alloc()
element = fppoly_from_network_input_poly(man , 0, 8, inf, sup, lpp_new, lexpr_cst, lexpr_dim, upp_new, uexpr_cst, uexpr_dim , 8)

rnn_handle_first_relu_layer(man, element, weights_l1, bias_l1, dim_l1, 2, 8, pre_l1)

rnn_handle_intermediate_relu_layer(man, element, weights_l2, bias_l2, dim_l2, 2, 8, pre_l2, True)

rnn_handle_intermediate_relu_layer(man, element, weights_l3, bias_l3, dim_l3, 2, 8, pre_l3, True)

rnn_handle_last_relu_layer(man, element, weights_op, bias_op, dim_op, 2, 8, pre_op, False, True)

elina_abstract0_fprint(cstdout, man, element, None)
'''

'''
#encode
# 2p1 + p2 + phi <= p1' <= p3+p4+2phi
# p1 + 2phi <= p2' <= p3 + p4 + phi
# p3 + 2p4 + phi <= p3' <= 4p4 
# p2 + p4 <= p4' <= p4 + 2phi

lexpr_weights = np.array([[2.0,1.0,1.0],[1.0,0.0,2.0],[1.0,2.0,1.0],[1.0,1.0,0.0]])
np.ascontiguousarray(lexpr_weights, dtype=np.double)
print("shape ",lexpr_weights.shape[0])
print("before ", lexpr_weights)
lpp = (lexpr_weights.__array_interface__['data'][0]+ np.arange(lexpr_weights.shape[0])*lexpr_weights.strides[0]).astype(np.uintp) 
lpp_new = np.array([2.0,1.0,1.0, 1.0,0.0,2.0, 1.0,2.0,1.0, 1.0,1.0,0.0])
lpp_new = np.ascontiguousarray(lpp_new, dtype=np.double)
print("weights ", lexpr_weights[0])

uexpr_weights = np.array([[1.0,1.0,2.0],[1.0,2.0,1.0],[0.0,0.0,4.0],[0.0,1.0,2.0]])
np.ascontiguousarray(uexpr_weights, dtype=np.double)
upp = (uexpr_weights.__array_interface__['data'][0]+ np.arange(uexpr_weights.shape[0])*uexpr_weights.strides[0]).astype(np.uintp)
upp_new = np.array([1.0,1.0,2.0, 1.0,2.0,1.0, 0.0,0.0,4.0, 0.0,1.0,2.0])
upp_new = np.ascontiguousarray(upp_new, dtype=np.double)

lexpr_cst = np.array([0.0,0.0,0.0,0.0])
np.ascontiguousarray(lexpr_cst, dtype=np.double)

uexpr_cst = np.array([0.0,0.0,0.0,0.0])
np.ascontiguousarray(uexpr_cst, dtype=np.double)

lexpr_dim = np.array([[0,1,4],[0,1,4],[2,3,4],[1,3,4]])
lexpr_dim = (lexpr_dim.__array_interface__['data'][0]+ np.arange(lexpr_dim.shape[0])*lexpr_dim.strides[0]).astype(np.uintp)

uexpr_dim = np.array([[2,3,4],[2,3,4],[0,1,3],[1,3,4]]) 
uexpr_dim = (uexpr_dim.__array_interface__['data'][0]+ np.arange(uexpr_dim.shape[0])*uexpr_dim.strides[0]).astype(np.uintp)

lexpr_size = np.array([3,3,3,3])
lexpr_size = lexpr_size.astype(np.uintp)
np.ascontiguousarray(lexpr_size, dtype=np.uintp)

uexpr_size = np.array([3,3,3,3])
uexpr_size = uexpr_size.astype(np.uintp)
np.ascontiguousarray(uexpr_size, dtype=np.uintp)


#print("type ", type(lexpr_size))

#encode
# 0.1 <= p1 <= 0.2
# 0.2 <= p2 <= 0.3
# 0.3 <= p3 <= 0.4
# 0.4 <= p4 <= 0.5
# 0.001 <= phi <= 0.002

inf = np.array([0.1,0.2,0.3,0.4,0.001])
sup = np.array([0.2,0.3,0.4,0.5,0.002])


man = fppoly_manager_alloc() 
#element = fppoly_from_network_input_poly(man,0,4,inf,sup, lpp, lexpr_cst, lexpr_dim, lexpr_size, upp, uexpr_cst, uexpr_dim ,uexpr_size)
element = fppoly_from_network_input_poly(man,0,4,inf,sup, lpp_new, lexpr_cst, lexpr_dim, upp_new, uexpr_cst, uexpr_dim ,3)

#handle first layer
weights = np.array([[1.0,-1.0,1.0,-1.0],[-1.0,1.0,-1.0,1.0],[1.0,1.0,-1.0,-1.0],[-1.0,-1.0,1.0,1.0]])
np.ascontiguousarray(weights, dtype=np.double)
weights = (weights.__array_interface__['data'][0]+ np.arange(weights.shape[0])*weights.strides[0]).astype(np.uintp) 


biases = np.array([1.0,-1.0,-1.0,1.0])
np.ascontiguousarray(biases, dtype = np.double)

pre0 = np.array([0, 0])
np.ascontiguousarray(pre0, dtype = ctypes.c_ulong)


pre = (c_size_t * 1)()
#pre[0] = None
pre_ptr = ctypes.cast(pre, ctypes.POINTER(ctypes.c_ulong))

sizeio = c_size_t()

#pre1 = c_int64()
#pre1_ptr = ctypes.cast(pre1, ctypes.POINTER(ctypes.c_ulong))

#ffn_handle_first_relu_layer(man,element,weights,biases,4, 4, ctypes.cast(pre, ctypes.POINTER(ctypes.c_ulong)))
ffn_handle_first_relu_layer(man, element, weights, biases, sizeio, sizeio, None)

elina_abstract0_fprint(cstdout,man,element,None)
'''