import numpy as np
from fppoly import *
from elina_abstract0 import *
from elina_manager import *
import csv
from utils import *

from read_net_file import *


'''
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)
'''

def get_expr_coeff(man, element, hidden_size, timestep, step_size, output_size, dimension, is_lower_only):

    if is_lower_only:
        l_outneuron_coef = []
        for op_idx in range(0, output_size):
            l_i = []
            lexpr = get_lexpr_for_output_neuron_simple(man, element, op_idx)
            for dim_idx in range(0, dimension):
                l_i.append(lexpr[0].sup_coeff[dim_idx])
            l_i.append(lexpr[0].sup_cst)
            l_outneuron_coef.append(l_i)
        lexpr_coeff = reorganize_output(l_outneuron_coef, hidden_size, timestep, step_size, output_size)
        #print(lexpr_coeff)
        return lexpr_coeff, None

    else:
        l_outneuron_coef = []
        u_outneuron_coef = []

        for op_idx in range(0, output_size):
            l_i = []
            u_i = []
            lexpr = get_lexpr_for_output_neuron_simple(man, element, op_idx)
            uexpr = get_uexpr_for_output_neuron_simple(man, element, op_idx)
            for dim_idx in range(0, dimension):
                l_i.append(lexpr[0].sup_coeff[dim_idx])
                u_i.append(uexpr[0].sup_coeff[dim_idx])
            l_i.append(lexpr[0].sup_cst)
            u_i.append(uexpr[0].sup_cst)
            l_outneuron_coef.append(l_i)
            u_outneuron_coef.append(u_i)

        lexpr_coeff = reorganize_output(l_outneuron_coef, hidden_size, timestep, step_size, output_size)
        uexpr_coeff = reorganize_output(u_outneuron_coef, hidden_size, timestep, step_size, output_size)

        #print(lexpr_coeff)
        #print(uexpr_coeff)

        return lexpr_coeff, uexpr_coeff