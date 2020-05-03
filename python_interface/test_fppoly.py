import numpy as np
from fppoly import *
from elina_abstract0 import *
from elina_manager import *

from ctypes import *
from ctypes.util import *

libc = CDLL(find_library('c'))
cstdout = c_void_p.in_dll(libc, '__stdoutp')

# test rnn

# input layer
'''
inf = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
sup = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
'''
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
weights_out = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
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

elina_abstract0_fprint(cstdout, man, element, None)

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