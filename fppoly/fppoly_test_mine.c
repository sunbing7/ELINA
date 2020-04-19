//
// Created by Bing Sun on 16/4/20.
//
#include <stdio.h>
#include "fppoly.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("The test requires two positive integers: (a) Number of variables and (b) Number of constraints");
        return 0;
    }
    printf("sunbing test");

    elina_manager_t *man;
    man = fppoly_manager_alloc();
#if 1
    size_t input_size = 2;
    double inf[2] = {-1.0, -1.0};
    double sup[2] = {1.0, 1.0};
    double lpp[2][2] = {{1, 0}, {0, 1}};
    double upp[2][2] =  {{1, 0}, {0, 1}};
    double lexpr_cst[2] = {0.0,0.0};
    double uexpr_cst[2] = {0.0,0.0};
    size_t lexpr_dim[2][2] = {{0,1}, {0,1}};
    size_t uexpr_dim[2][2] = {{0,1}, {0,1}};

    size_t lexpr_size[4] = {2, 2};

    double weights[2][2] = {{1.0,1.0}, {1.0,-1.0}};
    double bias[2] = {0, 0};

    double oweights[2][2] = {{1.0,1.0}, {0.0, 1.0}};
    double obias[2] = {1, 0};

    size_t predecessor[1] = {};
    size_t predecessor_l1[1] = {1};
    size_t predecessor_l2[1] = {2};
    size_t predecessor_l3[1] = {3};
    size_t predecessor_l4[1] = {4};
    size_t predecessor_l5[1] = {5};
    size_t predecessor_l6[1] = {6};

    /*
     * tet_weights should contain pointers points to weight arraies
     */
    volatile double * test_weights[2];

    for (int i = 0; i < 2; i++) {
        test_weights[i] = &(weights[i]);
    }

    volatile double * out_weights[2];

    for (int i = 0; i < 2; i++) {
        out_weights[i] = &(oweights[i]);
    }
#else
    size_t input_size = 4;
    double inf[5] = {0.1,0.2,0.3,0.4,0.001};
    double sup[5] = {0.2,0.3,0.4,0.5,0.002};
    double lpp[4][3] = {{2.0,1.0,1.0}, {1.0,0.0,2.0}, {1.0,2.0,1.0}, {1.0,1.0,0.0}};
    double upp[4][3] = {{1.0,1.0,2.0}, {1.0,2.0,1.0}, {0.0,0.0,4.0}, {0.0,1.0,2.0}};
    double lexpr_cst[4] = {0.0,0.0,0.0,0.0};
    double uexpr_cst[4] = {0.0,0.0,0.0,0.0};
    size_t lexpr_dim[4][3] = {{0,1,4}, {0,1,4}, {2,3,4}, {1,3,4}};
    size_t uexpr_dim[4][3] = {{2,3,4}, {2,3,4}, {0,1,3}, {1,3,4}};

    size_t lexpr_size[4] = {3,3,3,3};

    double weights[4][4] = {{1.0,-1.0,1.0,-1.0}, {-1.0,1.0,-1.0,1.0}, {1.0,1.0,-1.0,-1.0}, {-1.0,-1.0,1.0,1.0}};
    double bias[4] = {1.0,-1.0,-1.0,1.0};
    size_t predecessor[1] = {};
    size_t predecessor_l1[1] = {1};
    size_t predecessor_l2[1] = {2};
    size_t predecessor_l3[1] = {3};
    size_t predecessor_l4[1] = {4};
    size_t predecessor_l5[1] = {5};
    size_t predecessor_l6[1] = {6};


    /*
     * tet_weights should contain pointers points to weight arraies
     */
    volatile double * test_weights[4];

    for (int i = 0; i < 4; i++) {
        test_weights[i] = &(weights[i]);
    }
#endif
    elina_abstract0_t* element;
#if 1
    /* build abstract element of input layer from: inf, sup, lpp, upp, ldim, udim, lcst, ucst */
    element = fppoly_from_network_input_poly(man, 0, input_size, inf, sup,
                                             lpp, lexpr_cst, lexpr_dim, upp,
                                             uexpr_cst, uexpr_dim, lexpr_size[0]);
    ffn_handle_first_relu_layer(man, element, test_weights, bias,   input_size, input_size, predecessor);


    ffn_handle_intermediate_relu_layer(man, element, test_weights, bias,  input_size, input_size, predecessor_l1, false);


    ffn_handle_last_relu_layer(man, element, out_weights, obias,  input_size, input_size, predecessor_l2, false, false);

    // for debug purpose only
    ffn_handle_first_relu_layer(man, element, test_weights, bias,   input_size, input_size, predecessor);
#endif
}