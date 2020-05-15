import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import re

inname = '/Users/bing.sun/workspace/elina/ELINA-master-python/rnn_verifier/data/test/coeff_out_0.02.csv'

def extract_image_no(text):
    image_no = ''
    m = re.search('test image (.+?)', text)

    if m:
        image_no = m.group(1)
    return np.int(image_no)

def parseVec(net):
    return np.array(eval(net.readline()[:-1]))

def plot(file, out_size, dim):
    coef_l = []
    coef_u = []
    coef_l_p = []
    coef_u_p = []
    x_axis = []
    file_in = open(file,'r')
    for idx in range(0, dim):
        x_axis.append(idx)
    while True:
        curr_line = file_in.readline()[:-1]
        coef_l_i = []
        coef_u_i = []
        coef_l_p_i = []
        coef_u_p_i = []
        if 'test image' in curr_line:
            image_no = extract_image_no(curr_line)
            curr_line = file_in.readline()[:-1]  #'lower'

            for i in range (0, out_size):
                coef_l_i.append(parseVec(file_in))

            curr_line = file_in.readline()[:-1]  #'upper'

            for i in range (0, out_size):
                coef_u_i.append(parseVec(file_in))

            curr_line = file_in.readline()[:-1]  # 'lower perturbed'

            for i in range(0, out_size):
                coef_l_p_i.append(parseVec(file_in))

            curr_line = file_in.readline()[:-1]  # 'upper perturbed'

            for i in range(0, out_size):
                coef_u_p_i.append(parseVec(file_in))


            #debug

            for neuron in range (0, out_size):
                fig, ax = plt.subplots()
                ax.set_title('neuron {}'.format(neuron), color='C0')
                ax.plot(x_axis, coef_l_p_i[neuron], color='green', label='lexpr', alpha=0.5)
                ax.plot(x_axis, coef_u_p_i[neuron], color='darkorange',  label='uexpr', alpha=0.5)
                ax.legend()
                plt.show()

            coef_l.append(coef_l_i)
            coef_u.append(coef_u_i)
            coef_l_p.append(coef_l_p_i)
            coef_u_p.append(coef_u_p_i)


    fig, ax = plt.subplots()

    ax.plot(x_axis, coef_l[0][0], 'C1', label='lexpr')
    ax.plot(x_axis, coef_u[0][0], 'C2', label='uexpr')
    ax.legend()
    plt.show()

def plot_result():

    epsolion = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    ours = [1,	1,	0.976744186,	0.976744186,	0.965116279,	0.895348837,	0.790697674,	0.558139535,	0.395348837]
    deeppoly = [1,	1,	1,	1,	0.95,	0.75,	0.45,	0.22,	0.08]
    popqorn = 0.0131


    fig = plt.figure(num=3, figsize=(8, 5))
    plt.xlabel('Epsilon')
    plt.ylabel('Verfied Robustness')
    plt.plot(epsolion, ours,
             color='red',
             marker='o',
             label='Ours'
             )
    plt.plot(epsolion, deeppoly,
             color='green',
             linewidth=1.0,
             marker='^',
             label='DeepPoly'
             )

    plt.vlines(0.0131, 0, 1, color='darkorange', linestyle=':', label='POPQORN avg tolerance', data=None)
    plt.legend()
    plt.show()

    return

#plot(inname, 10, 849)
plot_result()