//
// Created by Bing Sun on 16/4/20.
//
#include <stdio.h>
#include "fppoly.h"
#if HAS_RNN
#define HAS_DEEPPOLY_EXAMPLE    0
#define HAS_FPPOLY_PY_EXAMPLE   0
#define HAS_RNN_EXAMPLE         1
#endif

int main(int argc, char **argv) {
    elina_abstract0_t* element;
    elina_manager_t *man;
    size_t input_size;
    man = fppoly_manager_alloc();
#if HAS_RNN_EXAMPLE
    /*
     *
     * evaluate expression: *(fppoly_t*)(element->value)
     */

    /*
     *  output layer
     */

    double weights_out[2][8] = {
            {1, 0, 0, 0, 0, 0, 0, 0},
            {-1, 1, 0, 0, 0, 0, 0, 0}
    };
    double bias_out[2] = {0, -2};

    double * weights_out_ptr[2];

    for (int i = 0; i < 2; i++) {
        weights_out_ptr[i] = &(weights_out[i]);
    }

    size_t expr_dim_op[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    size_t predecessor_out[1] = {3};

    size_t input_size_out = 2;
    size_t output_size_out = 2;
    /*
     * layer 3:
     */

    double weights_l3[2][8] = {
            {0.1, -0.1, 1.0, 1.0, 0, 0, 0, 0},
            {-0.1, 0.1, 1.0, -1.0, 0, 0, 0, 0}
    };

    double bias_l3[2] = {0, 0};

    size_t expr_dim_l3[2][8] = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    double * weights_l3_ptr[2];

    for (int i = 0; i < 2; i++) {
        weights_l3_ptr[i] = &(weights_l3[i]);
    }
#if 0
    double * expr_dim_l3_ptr[2];

    for (int i = 0; i < 2; i++) {
        expr_dim_l3_ptr[i] = &(expr_dim_l3[i]);
    }
#else
    double * expr_dim_l3_ptr = &(expr_dim_l3[0]);
#endif
    size_t predecessor_l3[1] = {3};

    size_t input_size_l3 = 4;
    size_t output_size_l3 = 2;

    /*
     * layer 2:
     */

    double weights_l2[2][8] = {
            {0.1, -0.1, 0, 0, 1.0, 1.0, 0, 0},
            {-0.1, 0.1, 0, 0, 1.0, -1.0, 0, 0}
    };

    double bias_l2[2] = {0, 0};

    size_t expr_dim_l2[2][8] = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    double * weights_l2_ptr[2];

    for (int i = 0; i < 2; i++) {
        weights_l2_ptr[i] = &(weights_l2[i]);
    }
#if 0
    double * expr_dim_l2_ptr[2];

    for (int i = 0; i < 2; i++) {
        expr_dim_l2_ptr[i] = &(expr_dim_l2[i]);
    }
#else
    double * expr_dim_l2_ptr = &(expr_dim_l2[0]);
#endif

    size_t predecessor_l2[1] = {2};

    size_t input_size_l2 = 4;
    size_t output_size_l2 = 2;

    /*
     * layer 1:
     */

    double weights_l1[2][8] = {
            {0.1, -0.1, 0, 0, 0, 0, 1.0, 1.0},
            {-0.1, 0.1, 0, 0, 0, 0, 1.0, -1.0}
    };

    double bias_l1[2] = {0, 0};

    size_t expr_dim_l1[2][8] = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    double * weights_l1_ptr[2];

    for (int i = 0; i < 2; i++) {
        weights_l1_ptr[i] = &(weights_l1[i]);
    }
#if 0
    double * expr_dim_l1_ptr[2];

    for (int i = 0; i < 2; i++) {
        expr_dim_l1_ptr[i] = &(expr_dim_l1[i]);
    }
#else
    double * expr_dim_l1_ptr =  &(expr_dim_l1[0]);
#endif
    size_t predecessor_l1[1] = {1};

    size_t input_size_l1 = 4;
    size_t output_size_l1 = 2;

    /*
     * input
     */

    double inf[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    double sup[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double lpp[8][8] = {{1, 0, 0, 0, 0, 0, 0, 0},
                        {0, 1, 0, 0, 0, 0, 0, 0},
                        {0, 0, 1, 0, 0, 0, 0, 0},
                        {0, 0, 0, 1, 0, 0, 0, 0},
                        {0, 0, 0, 0, 1, 0, 0, 0},
                        {0, 0, 0, 0, 0, 1, 0, 0},
                        {0, 0, 0, 0, 0, 0, 1, 0},
                        {0, 0, 0, 0, 0, 0, 0, 1},
    };
    double upp[8][8] = {{1, 0, 0, 0, 0, 0, 0, 0},
                        {0, 1, 0, 0, 0, 0, 0, 0},
                        {0, 0, 1, 0, 0, 0, 0, 0},
                        {0, 0, 0, 1, 0, 0, 0, 0},
                        {0, 0, 0, 0, 1, 0, 0, 0},
                        {0, 0, 0, 0, 0, 1, 0, 0},
                        {0, 0, 0, 0, 0, 0, 1, 0},
                        {0, 0, 0, 0, 0, 0, 0, 1},
    };

    double lexpr_cst[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double uexpr_cst[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    size_t lexpr_dim[8][8] = {{0, 1, 2, 3, 4, 5, 6, 7},
                              {0, 1, 2, 3, 4, 5, 6, 7},
                              {0, 1, 2, 3, 4, 5, 6, 7},
                              {0, 1, 2, 3, 4, 5, 6, 7},
                              {0, 1, 2, 3, 4, 5, 6, 7},
                              {0, 1, 2, 3, 4, 5, 6, 7},
                              {0, 1, 2, 3, 4, 5, 6, 7},
                              {0, 1, 2, 3, 4, 5, 6, 7}
    };
    size_t uexpr_dim[8][8] = {{0, 1, 2, 3, 4, 5, 6, 7},
                              {0, 1, 2, 3, 4, 5, 6, 7},
                              {0, 1, 2, 3, 4, 5, 6, 7},
                              {0, 1, 2, 3, 4, 5, 6, 7},
                              {0, 1, 2, 3, 4, 5, 6, 7},
                              {0, 1, 2, 3, 4, 5, 6, 7},
                              {0, 1, 2, 3, 4, 5, 6, 7},
                              {0, 1, 2, 3, 4, 5, 6, 7}
    };

    size_t lexpr_size[8] = {8, 8, 8, 8, 8, 8, 8, 8};

    size_t predecessor[1] = {-1};




    /* build abstract element of input layer from: inf, sup, lpp, upp, ldim, udim, lcst, ucst */
    element = fppoly_from_network_input_poly(man, 0, 8, inf, sup,
                                             lpp, lexpr_cst, lexpr_dim, upp,
                                             uexpr_cst, uexpr_dim, lexpr_size[0]);
    ffn_handle_first_relu_layer_(man, element, weights_l1_ptr, bias_l1, expr_dim_l1_ptr, 2, 8, predecessor);


    ffn_handle_intermediate_relu_layer_(man, element, weights_l2_ptr, bias_l2, expr_dim_l2_ptr, 2, 8, predecessor_l1, true);

    ffn_handle_intermediate_relu_layer_(man, element, weights_l3_ptr, bias_l3, expr_dim_l3_ptr, 2, 8, predecessor_l2, true);

    ffn_handle_last_relu_layer_(man, element, weights_out_ptr, bias_out, expr_dim_op, 2, 8, predecessor_out, false, true);

    // for debug purpose only
    ffn_handle_intermediate_relu_layer_(man, element, weights_l3_ptr, bias_l3, expr_dim_l3_ptr, output_size_l1, 8, predecessor_l2, true);
    //ffn_handle_first_relu_layer(man, element, test_weights, bias,   input_size, input_size, predecessor);

#endif

#if HAS_DEEPPOLY_EXAMPLE
    /*
     * DeepPoly Paper Example
     */
    input_size = 2;
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

    /* build abstract element of input layer from: inf, sup, lpp, upp, ldim, udim, lcst, ucst */
    element = fppoly_from_network_input_poly(man, 0, input_size, inf, sup,
                                             lpp, lexpr_cst, lexpr_dim, upp,
                                             uexpr_cst, uexpr_dim, lexpr_size[0]);
    ffn_handle_first_relu_layer(man, element, test_weights, bias,   input_size, input_size, predecessor);


    ffn_handle_intermediate_relu_layer(man, element, test_weights, bias,  input_size, input_size, predecessor_l1, false);


    ffn_handle_last_relu_layer(man, element, out_weights, obias,  input_size, input_size, predecessor_l2, false, false);

    // for debug purpose only
    ffn_handle_first_relu_layer(man, element, test_weights, bias,   input_size, input_size, predecessor);
#endif    /* HAS_DEEPPOLY_EXAMPLE */
#if HAS_FPPOLY_PY_EXAMPLE
    input_size = 4;
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

    /* build abstract element of input layer from: inf, sup, lpp, upp, ldim, udim, lcst, ucst */
    element = fppoly_from_network_input_poly(man, 0, input_size, inf, sup,
                                                      lpp, lexpr_cst, lexpr_dim, upp,
                                             uexpr_cst, uexpr_dim, lexpr_size[0]);
    ffn_handle_first_relu_layer(man, element, test_weights, bias,   input_size, input_size, predecessor);


    ffn_handle_intermediate_relu_layer(man, element, test_weights, bias,  input_size, input_size, predecessor_l1, false);


    ffn_handle_last_relu_layer(man, element, out_weights, obias,  input_size, input_size, predecessor_l2, false, false);

    // for debug purpose only
    ffn_handle_first_relu_layer(man, element, test_weights, bias,   input_size, input_size, predecessor);
#endif  /* HAS_FPPOLY_PY_EXAMPLE */
}