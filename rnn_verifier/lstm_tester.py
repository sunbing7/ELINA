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
import math


'''
lstm calculator

lstm_layer(hx, ci, weights_ip, bias_ip, hidden_size)
ret: ct, ht

dense_op(x, w, b, out_size)
ret: out


'''
def sigmoid(x):
    try:
        out = 1 / (1 + math.exp(-x))
    except OverflowError:
        out = 0
    return out

def lstm_cell(hx, ci, wi, wc, wf, wo, bi, bc, bf, bo):
    f = sigmoid(np.matmul(hx.transpose(), wf) + bf)
    i = sigmoid(np.matmul(hx.transpose(), wi) + bi)
    c = np.tanh(np.matmul(hx.transpose(), wc) + bc)
    o = sigmoid(np.matmul(hx.transpose(), wo) + bo)

    ct = ci * f + i * c

    ht = o * np.tanh(ct)
    return ct, ht


'''
x: x1, x2, x3

'''
def dense_op(x, w, b, out_size):
    out = []
    for i in range (0, out_size):
        out_i = np.matmul(w[i], np.array(x).transpose()) + b[i]
        out.append(out_i)
    print('out: {}'.format(out))
    return out


def demo():
    W_i = np.array([[0.25, 0.5], [0.25, 0.5]])
    W_c = np.array([[0.4, 0.3], [0.4, 0.3]])
    W_f = np.array([[0.06, 0.03], [0.06, 0.03]])
    W_o = np.array([[0.04, 0.02], [0.04, 0.02]])

    c_i = np.array([0, 0])
    hx_in = np.array([0.06, 0.3])

    c, h = lstm_cell(hx_in, c_i, W_i, W_c, W_f, W_o, 0.01, 0.05, 0.002, 0.001)
    print('ht: {}, ct: {}'.format(h, c))



'''
hx: h1 h2 ... x1 x2 x3 ...
'''
def lstm_layer(hx, ci, weights_ip, bias_ip, hidden_size):
    ct = []
    ht = []
    for i in range (0,  hidden_size):

        # !!!!!! weight organization of Pytorch !!!!!!!
        W_i = weights_ip[i]
        W_f = weights_ip[i + hidden_size]
        W_c = weights_ip[i + 2 * hidden_size]
        W_o = weights_ip[i + 3 * hidden_size]

        c_i = ci[i]
        hx_in = hx
        bi = bias_ip[i]
        bf = bias_ip[i + hidden_size]
        bc = bias_ip[i + 2 * hidden_size]
        bo = bias_ip[i + 3 * hidden_size]
        '''
        W_i = weights_ip[i * 4]
        W_f = weights_ip[i * 4 + 1]
        W_c = weights_ip[i * 4 + 2]
        W_o = weights_ip[i * 4 + 3]

        c_i = ci[i]
        hx_in = hx
        bi = bias_ip[i * 4]
        bf = bias_ip[i * 4 + 1]
        bc = bias_ip[i * 4 + 2]
        bo = bias_ip[i * 4 + 3]
        '''

        c, h = lstm_cell(hx_in, c_i, W_i, W_c, W_f, W_o, bi, bc, bf, bo)
        ct.append(c)
        ht.append(h)
    #print('ct: {}, ht: {}'.format(ct, ht))
    return ct, ht

def test():

    W = np.array([[0.1, -0.1, 1.0], [0.1, -0.1, 1.0], [0.1, -0.1, 1.0], [0.1, -0.1, 1.0],
                  [-0.1, 0.1, -1.0], [-0.1, 0.1, -1.0], [-0.1, 0.1, -1.0], [-0.1, 0.1, -1.0]])

    # c1 c2
    c_i = np.array([0, 1])

    # h1 h2 x1
    hx_in = np.array([1.0, 1.0, 1.0])

    b = np.array([0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0])
    c, h = lstm_layer(hx_in, c_i, W, b, 2)
    print('layer 1: ct: {}, ht: {}'.format(c, h))

    hx_in_2 = np.append(h, 1.0)
    c, h = lstm_layer(hx_in_2, c, W, b, 2)
    print('layer 2: ct: {}, ht: {}'.format(c, h))

    w_o = np.array([[1.0, 0.1], [0.1, -1]])
    b_o = np.array([0.0, 0.0])

    out = dense_op(h, w_o, b_o, 2)
    print('out: {}'.format(out))



#test()
'''
wip = np.array([-0.470066,
0.400239,
0.670059,
-0.162375,
-0.427433,
0.122532,
0.71595,
0.00543035,
0.52126,
0.0,
0.0133846,
-0.443283,
-0.000444134,
-0.866986,
0.705446,
-0.422642])

weight = np.array([[0.17022517,	0.38381132,	1.5280861,	-0.9514082,	1.7871335,	0.11251447,	0.49984866,	-0.6630708,	-1.0509821,	0.8586154,	1.0123733,	1.1181756,	0.816764,	-1.0512881,	0.6376488,	-0.85311913],
[-1.2996155,	-1.1761786,	-0.65476096,	-0.4888934,	1.1801509,	-0.7007094,	0.31861016,	1.0736929,	0.4141738,	-1.2664533,	-1.2968868,	-0.3618296,	0.608466,	0.57081395,	0.6518062,	-1.250568],
[0.6375567,	0.725146,	-0.6224704,	1.7212656,	-0.1733351,	0.9808656,	0.23222463,	1.1115428,	-0.35020334,	1.0062854,	-1.4898971,	1.0481012,	1.0044116,	0.50731426,	-0.7452904,	-1.0026991],
[-0.15447581,	0.91999954,	-0.6591229,	-1.0231088,	-1.282357,	0.38802376,	-1.2649097,	-0.4239541,	-0.16787446,	0.968638,	-0.93254584,	-1.0794476,	-0.6260482,	0.7664414,	0.556338,	-1.0434152],
[0.9030005,	0.21334352,	0.23407894,	0.8399514,	1.607096,	0.982075,	-0.66545665,	-0.76966363,	0.77312046,	-1.3156096,	0.85691863,	-0.26941273,	-1.5896572,	-0.6920342,	-1.1301818,	0.6771621],
[0.47654426,	-0.94451535,	-0.77919567,	-0.68963903,	-1.242629,	1.0721662,	0.30808362,	-0.5538163,	-1.7058523,	-0.44963646,	-0.5614144,	-1.1229217,	0.84109014,	-1.6121004,	0.79513544,	-0.00259185],
[-0.6215877,	0.8817974,	0.4816785,	1.5123652,	0.11644202,	-0.5778217,	1.0821314,	0.91501564,	-1.3766387,	-1.055394,	1.2268212,	-0.5759099,	0.22745717,	0.6587185,	0.07503752,	0.6948222],
[-1.3152026,	-0.6803731,	1.3324262,	-0.3537254,	-0.7618102,	-0.8338717,	-1.2043108,	0.8270263,	1.2856427,	0.52396107,	-1.2541578,	0.9658531,	-1.4776226,	-0.5038267,	-0.8960482,	0.6126227],
[0.7992754,	-0.5650222,	-0.89565873,	0.36233515,	-0.38491416,	0.3365784,	1.026035,	-0.8530534,	1.0263965,	0.9674349,	-0.24626292,	-0.6138685,	0.53302765,	1.2719301,	0.92140496,	0.6123375],
[0.7609211,	-0.46110487,	-0.51536435,	-1.4453771,	-1.0353745,	-0.83643246,	-0.69077516,	-1.2131653,	0.7545125,	-0.04476993,	0.880509,	0.16893655,	-1.9007115,	-0.07228827,	-0.87830484,	0.5810915,]])

bias = np.array([-0.45091888,
-0.15766275,
0.06207726,
-0.24274458,
0.10944786,
0.003621922,
-0.120310485,
0.62626487,
-0.26801372,
0.09437538])

out = np.matmul(weight, wip) + bias
print(out)
'''