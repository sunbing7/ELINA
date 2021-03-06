//
// Created by Bing Sun on 16/4/20.
//
#include <stdio.h>
#include "fppoly.h"
#include "lstm_approx.c"

#if HAS_RNN
#define HAS_DEEPPOLY_EXAMPLE    0
#define HAS_FPPOLY_PY_EXAMPLE   0
#define HAS_RNN_EXAMPLE         0
#define HAS_RNN_DEMO            0
#define HAS_LSTM_DEMO           0
#define HAS_RNN_TANH            1
#endif

int main(int argc, char **argv) {
    elina_abstract0_t* element;
    elina_manager_t *man;
    size_t input_size;
    man = fppoly_manager_alloc();
#if HAS_LSTM_DEMO


#define our_test    1
#define Their_test  0
#define value_test  0
 /*
  * input
  */
#if value_test
    #include "matrix.h"
    double lpp[DIMENSTION][DIMENSTION];
    double expr_cst[DIMENSTION];
    size_t expr_dim[DIMENSTION][DIMENSTION];
    size_t expr_size[DIMENSTION];

    //sunbing
    uint8_t volatile test_buf[4][592];

    size_t volatile *dim0_temp = (size_t*)&test_buf[0];
    size_t volatile *dim1_temp = (size_t*)&test_buf[1];
    size_t volatile *dim2_temp = (size_t*)&test_buf[2];
    size_t volatile *dim3_temp = (size_t*)&test_buf[3];

    for (int i = 0; i < DIMENSTION; i++) {
        for (int j = 0; j < DIMENSTION; j++) {
            if (i == j) {
                lpp[i][j] = 1.0;
            } else {
                lpp[i][j] = 0.0;
            }
            expr_dim[i][j] = (uint64_t)j;
        }
        expr_cst[i] = 0.0;
        expr_size[i] = DIMENSTION;
    }

    /* layer 1 */
    double * weights_ptr[HIDDEN_GATE];
    for (int i = 0; i < HIDDEN_GATE; i++)
    {
        weights_ptr[i] = &(W_net[i]);
    }

    /* layer 2 */
    double * weights_ptr2[HIDDEN_GATE];
    for (int i = 0; i < HIDDEN_GATE; i++)
    {
        weights_ptr2[i] = &(W_net2[i]);
    }


    /* layer 3 */
    double * weights_ptr3[HIDDEN_GATE];
    for (int i = 0; i < HIDDEN_GATE; i++)
    {
        weights_ptr3[i] = &(W_net3[i]);
    }


    /* layer 4 */
    double * weights_ptr4[HIDDEN_GATE];
    for (int i = 0; i < HIDDEN_GATE; i++)
    {
        weights_ptr4[i] = &(W_net4[i]);
    }


    /* layer 5 */
    double * weights_ptr5[HIDDEN_GATE];
    for (int i = 0; i < HIDDEN_GATE; i++)
    {
        weights_ptr5[i] = &(W_net5[i]);
    }


    /* layer 6 */
    double * weights_ptr6[HIDDEN_GATE];
    for (int i = 0; i < HIDDEN_GATE; i++)
    {
        weights_ptr6[i] = &(W_net6[i]);
    }


    /* layer 7 */
    double * weights_ptr7[HIDDEN_GATE];
    for (int i = 0; i < HIDDEN_GATE; i++)
    {
        weights_ptr7[i] = &(W_net7[i]);
    }



    double * in_ptr[DIMENSTION];
    for (int i = 0; i < DIMENSTION; i++) {
        in_ptr[i] = &(lpp[i]);
    }


    double * weights_ptr_op[OUT_CLASS];
    for (int i = 0; i < OUT_CLASS; i++) {
        weights_ptr_op[i] = &weight_op[i];
    }

    double weights_l[OUT_CLASS * (OUT_CLASS - 1)][DIMENSTION];
    double bias_l[OUT_CLASS * (OUT_CLASS - 1)];

    int w_idx = 0;
    for (int y1_idx = 0; y1_idx < OUT_CLASS; y1_idx++) {
        for (int y2_idx = 0; y2_idx < OUT_CLASS; y2_idx++) {
            if (y1_idx == y2_idx)
                continue;
            bias_l[w_idx] = 0.0;
            /* get compare coefficient */
            for (int i = 0; i < DIMENSTION; i++) {
                if (i == y1_idx)
                    weights_l[w_idx][i] = 1.0;
                else if (i == y2_idx)
                    weights_l[w_idx][i] = -1.0;
                else
                    weights_l[w_idx][i] = 0.0;
            }
            w_idx++;
        }
    }



    double * weights_l_ptr[OUT_CLASS * (OUT_CLASS - 1)];

    for (int i = 0; i < OUT_CLASS * (OUT_CLASS - 1); i++) {
        weights_l_ptr[i] = &(weights_l[i]);
    }

    size_t predecessor_l[1] = {TIMESTEP + 2};

    size_t predecessor[1] = {-1};
    size_t predecessor_l1[1] = {1};
    size_t predecessor_l2[1] = {2};
    size_t predecessor_l3[1] = {3};
    size_t predecessor_l4[1] = {4};
    size_t predecessor_l5[1] = {5};
    size_t predecessor_l6[1] = {6};
    size_t predecessor_l7[1] = {7};
    size_t predecessor_o[1] = {TIMESTEP + 1};

    // (*(fppoly_t*)(element->value))->layers[0]->neurons[7]

    element = fppoly_from_network_input_poly(man, 0, DIMENSTION, inf, sup, lpp, expr_cst, expr_dim, lpp, expr_cst, expr_dim, expr_size[0]);

    ffn_handle_first_mul_layer_(man, element, in_ptr, ip_bias, expr_dim,  DIMENSTION, predecessor);

    lstm_handle_intermediate_layer_(man, element, weights_ptr, bias_h, expr_dim, DIMENSTION, HIDDEN_LAYER, predecessor_l1, true);
    lstm_handle_intermediate_layer_(man, element, weights_ptr2, bias_h, expr_dim, DIMENSTION, HIDDEN_LAYER, predecessor_l2, true);
    lstm_handle_intermediate_layer_(man, element, weights_ptr3, bias_h, expr_dim, DIMENSTION, HIDDEN_LAYER, predecessor_l3, true);
    lstm_handle_intermediate_layer_(man, element, weights_ptr4, bias_h, expr_dim, DIMENSTION, HIDDEN_LAYER, predecessor_l4, true);
    lstm_handle_intermediate_layer_(man, element, weights_ptr5, bias_h, expr_dim, DIMENSTION, HIDDEN_LAYER, predecessor_l5, true);
    lstm_handle_intermediate_layer_(man, element, weights_ptr6, bias_h, expr_dim, DIMENSTION, HIDDEN_LAYER, predecessor_l6, true);
    lstm_handle_intermediate_layer_(man, element, weights_ptr7, bias_h, expr_dim, DIMENSTION, HIDDEN_LAYER, predecessor_l7, true);

    lstm_handle_last_layer_(man, element, weights_ptr_op,  &bias_op, expr_dim, HIDDEN_LAYER, OUT_CLASS, predecessor_o, true);

    //compare classes
    lstm_handle_last_layer_(man, element, weights_l_ptr,  &bias_l, expr_dim, DIMENSTION, OUT_CLASS * (OUT_CLASS - 1), predecessor_l, true);

    //for debug only
    handle_lstm_layer_(man, element, weights_ptr,  bias_h, expr_dim, 3, 1, predecessor_l1, true);
#endif  /* value_test */
#if Their_test
    /*
  * input
  */

    uint32_t timestep = 3;

    double inf[3] = {0.0, 0.2, 0.1};
    double sup[3] = {0.0, 0.2, 0.1};
    double lpp[3][3] = {{1, 0, 0},
                        {0, 1, 0},
                        {0, 0, 1}
    };
    double upp[3][3] = {{1, 0, 0},
                        {0, 1, 0},
                        {0, 0, 1}
    };

    double lexpr_cst[3] = {0.0, 0.0, 0.0};
    double uexpr_cst[3] = {0.0, 0.0, 0.0};

    size_t lexpr_dim[3][3] = {{0, 1, 2},
                              {0, 1, 2},
                              {0, 1, 2}
    };
    size_t uexpr_dim[3][3] = {{0, 1, 2},
                              {0, 1, 2},
                              {0, 1, 2}
    };
    size_t lexpr_size[3] = {3, 3, 3};

    size_t predecessor[1] = {-1};





    double W_i[3] = {0.25, 0, 0.5};
    double W_c[3] = {0.4, 0, 0.3};
    double W_f[3] = {0.06, 0, 0.03};
    double W_o[3] = {0.04, 0, 0.02};


    double * weights_ptr[4];

    {
        weights_ptr[0] = &(W_i[0]);
        weights_ptr[1] = &(W_c[0]);
        weights_ptr[2] = &(W_f[0]);
        weights_ptr[3] = &(W_o[0]);
    }

    /*
     * layer 2
     */




    double W_i2[3] = {0.25, 0.5, 0.0};


    double W_c2[3] = {0.4, 0.3, 0.0};

    double W_f2[3] = {0.06, 0.03, 0.0};
    double W_o2[3] = {0.04, 0.02, 0.0};


    double * weights_ptr2[4];

    {
        weights_ptr2[0] = &(W_i2[0]);
        weights_ptr2[1] = &(W_c2[0]);
        weights_ptr2[2] = &(W_f2[0]);
        weights_ptr2[3] = &(W_o2[0]);
    }

    double bias[4] = {0.01, 0.05, 0.002, 0.001};
    double ip_bias[3] = {0.0, 0.0, 0.0};

    double * in_ptr[3];
    for (int i = 0; i < 3; i++) {
        in_ptr[i] = &(upp[i]);
    }

    double weight_op = 0.6;
    double bias_op = 0.025;

    double * weights_ptr_op[2];
    weights_ptr_op[0] = &weight_op;

    size_t predecessor_l1[1] = {0};
    size_t predecessor_l2[1] = {1};
    size_t predecessor_o[1] = {2};

    element = fppoly_from_network_input_poly(man, 0, 3, inf, sup,
                                             lpp, lexpr_cst, lexpr_dim, upp,
                                             uexpr_cst, uexpr_dim, lexpr_size[0]);

    ffn_handle_first_mul_layer_(man, element, in_ptr, ip_bias, lexpr_dim,  3, predecessor);

    //handle_lstm_layer_(man, element, weights_ptr,  bias, 8, 2, lexpr_dim, predecessor_l1, true);
    lstm_handle_intermediate_layer_(man, element, weights_ptr, bias, lexpr_dim, 3, 1, predecessor_l1, true);

    lstm_handle_intermediate_layer_(man, element, weights_ptr2, bias, lexpr_dim, 3, 1, predecessor_l2, true);

    lstm_handle_last_layer_(man, element, weights_ptr_op,  &bias_op, lexpr_dim, 1, 1, predecessor_o, true);
    //for debug only
    handle_lstm_layer_(man, element, weights_ptr,  bias, lexpr_dim, 3, 1, predecessor_l1, true);
#endif
#if our_test
    /*
      * input
      */

    uint32_t timestep = 3;

    double inf[8] = {1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0};
    double sup[8] = {1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0};
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

    /*
     * W_f, W_i, W_c, W_o
     */

    double W_f[2][8] = {{0.1, -0.1, 0, 0, 0, 0, 1, 1},
                        {-0.1, 0.1, 0, 0, 0, 0, 1, -1}};

    double b_f[2] = {0, 0};

    double W_i[2][8] = {{0.1, -0.1, 0, 0, 0, 0, 1, 1},
                        {-0.1, 0.1, 0, 0, 0, 0, 1, -1}};

    double b_i[2] = {0, 0};

    double W_c[2][8] = {{0.1, -0.1, 0, 0, 0, 0, 1, 1},
                        {-0.1, 0.1, 0, 0, 0, 0, 1, -1}};
    double b_c[2] = {0, 0};

    double W_o[2][8] = {{0.1, -0.1, 0, 0, 0, 0, 1, 1},
                        {-0.1, 0.1, 0, 0, 0, 0, 1, -1}};

    double b_o[2] = {0, 0};

    double * weights_ptr[8];

    for (int i = 0; i < 2; i++) {
        weights_ptr[i] = &(W_i[i]);
        weights_ptr[i + 2] = &(W_c[i]);
        weights_ptr[i + 4] = &(W_f[i]);
        weights_ptr[i + 6] = &(W_o[i]);
    }

    /*
     * layer 2
     */

    /*
     * W_f, W_i, W_c, W_o
     */
        double W_f2[2][8] = {{0.1, -0.1, 0, 0, 1, 1, 0, 0},
                        {-0.1, 0.1, 0, 0, 1, -1, 0, 0}};


    double W_i2[2][8] = {{0.1, -0.1, 0, 0, 1, 1, 0, 0},
                         {-0.1, 0.1, 0, 0, 1, -1, 0, 0}};


    double W_c2[2][8] = {{0.1, -0.1, 0, 0, 1, 1, 0, 0},
                         {-0.1, 0.1, 0, 0, 1, -1, 0, 0}};


    double W_o2[2][8] = {{0.1, -0.1, 0, 0, 1, 1, 0, 0},
                         {-0.1, 0.1, 0, 0, 1, -1, 0, 0}};


    double * weights_ptr2[8];

    for (int i = 0; i < 2; i++) {
        weights_ptr2[i] = &(W_f2[i]);
        weights_ptr2[i + 2] = &(W_f2[i]);
        weights_ptr2[i + 4] = &(W_f2[i]);
        weights_ptr2[i + 6] = &(W_f2[i]);
    }

    /*
     * layer 3
     */

    /*
     * W_f, W_i, W_c, W_o
     */
    double W_f3[2][8] = {{0.1, -0.1, 1, 1, 0, 0, 0, 0},
                         {-0.1, 0.1, 1, -1, 0, 0, 0, 0}};


    double W_i3[2][8] = {{0.1, -0.1, 1, 1, 0, 0, 0, 0},
                         {-0.1, 0.1, 1, -1, 0, 0, 0, 0}};


    double W_c3[2][8] = {{0.1, -0.1, 1, 1, 0, 0, 0, 0},
                         {-0.1, 0.1, 1, -1, 0, 0, 0, 0}};


    double W_o3[2][8] = {{0.1, -0.1, 1, 1, 0, 0, 0, 0},
                         {-0.1, 0.1, 1, -1, 0, 0, 0, 0}};


    double * weights_ptr3[8];

    for (int i = 0; i < 2; i++) {
        weights_ptr3[i] = &(W_f3[i]);
        weights_ptr3[i + 2] = &(W_i3[i]);
        weights_ptr3[i + 4] = &(W_c3[i]);
        weights_ptr3[i + 6] = &(W_o3[i]);
    }


    double bias[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double ip_bias[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    double * in_ptr[8];
    for (int i = 0; i < 8; i++) {
        in_ptr[i] = &(upp[i]);
    }

    double weight_op[2] = {{-10.0, 10.0}, {-5, 13}};
    double bias_op[2] = {0.0, 0.0};

    double * weights_ptr_op[2];
    weights_ptr_op[0] = &weight_op[0];
    weights_ptr_op[1] = &weight_op[1];

    size_t predecessor_l1[1] = {1};
    size_t predecessor_l2[1] = {2};
    size_t predecessor_l3[1] = {3};
    size_t predecessor_o[1] = {4};

    element = fppoly_from_network_input_poly(man, 0, 8, inf, sup,
                                             lpp, lexpr_cst, lexpr_dim, upp,
                                             uexpr_cst, uexpr_dim, lexpr_size[0]);

    //create_lstm_layer(man, element, 2, predecessor);
    ffn_handle_first_mul_layer_(man, element, in_ptr, ip_bias, lexpr_dim,  8, predecessor);

    //handle_lstm_layer_(man, element, weights_ptr,  bias, 8, 2, lexpr_dim, predecessor_l1, true);
    lstm_handle_intermediate_layer_(man, element, weights_ptr, bias, lexpr_dim, 8, 2, predecessor_l1, true);

    lstm_handle_intermediate_layer_(man, element, weights_ptr2, bias, lexpr_dim, 8, 2, predecessor_l2, true);

    lstm_handle_intermediate_layer_(man, element, weights_ptr3, bias, lexpr_dim, 8, 2, predecessor_l3, true);

    lstm_handle_last_layer_(man, element, weights_ptr_op,  &bias_op, lexpr_dim, 2, 2, predecessor_o, true);



    handle_lstm_layer_(man, element, weights_ptr,  bias, lexpr_dim, 8, 2, predecessor_l1, true);
#endif


#if 0
    ffn_handle_first_relu_layer_(man, element, weights_l1_ptr, bias_l1, expr_dim_l1_ptr, 2, 8, predecessor);


    ffn_handle_intermediate_relu_layer_(man, element, weights_l2_ptr, bias_l2, expr_dim_l2_ptr, 2, 8, predecessor_l1, true);

    ffn_handle_intermediate_relu_layer_(man, element, weights_l3_ptr, bias_l3, expr_dim_l3_ptr, 2, 8, predecessor_l2, true);

    ffn_handle_last_relu_layer_(man, element, weights_out_ptr, bias_out, expr_dim_op, 2, 8, predecessor_out, false, true);
#endif
#endif  /* HAS_LSTM_DEMO */
#if HAS_RNN_DEMO
    /*
     *
     * evaluate expression: *(fppoly_t*)(element->value)
     */

    /*
     *  output layer
     */

    double weights_out[2][11] = {
            {-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    };
    double bias_out[2] = {0, 0};

    double * weights_out_ptr[2];
    weights_out_ptr[0] = &(weights_out[0]);
    weights_out_ptr[1] = &(weights_out[1]);

    size_t expr_dim_op[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    size_t predecessor_out[1] = {3};

    /*
     * layer 3:
     */

    double weights_l3[2][11] = {
            {0.1, -0.1, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {-0.1, 0.1, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    };

    double bias_l3[2] = {0, 0};

    size_t expr_dim_l3[2][11] = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};

    double * weights_l3_ptr[2];

    for (int i = 0; i < 2; i++) {
        weights_l3_ptr[i] = &(weights_l3[i]);
    }

    double * expr_dim_l3_ptr = &(expr_dim_l3[0]);

    size_t predecessor_l3[1] = {3};

    size_t input_size_l3 = 4;
    size_t output_size_l3 = 2;

    /*
     * layer 2:
     */

    double weights_l2[2][11] = {
            {0.1, -0.1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0},
            {-0.1, 0.1, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0}
    };

    double bias_l2[2] = {0, 0};

    size_t expr_dim_l2[2][11] = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};

    double * weights_l2_ptr[2];

    for (int i = 0; i < 2; i++) {
        weights_l2_ptr[i] = &(weights_l2[i]);
    }

    double * expr_dim_l2_ptr = &(expr_dim_l2[0]);


    size_t predecessor_l2[1] = {2};

    size_t input_size_l2 = 4;
    size_t output_size_l2 = 2;

    /*
     * layer 1:
     */

    double weights_l1[2][11] = {
            {0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
            {-0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0}
    };

    double bias_l1[2] = {0, 0};

    size_t expr_dim_l1[2][11] = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};

    double * weights_l1_ptr[2];

    for (int i = 0; i < 2; i++) {
        weights_l1_ptr[i] = &(weights_l1[i]);
    }

    double * expr_dim_l1_ptr =  &(expr_dim_l1[0]);

    size_t predecessor_l1[1] = {1};

    size_t input_size_l1 = 4;
    size_t output_size_l1 = 2;

    /*
     * input
     */

    //double inf[8] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double sup[11] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double inf[11] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    double lpp[11][11] = {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
    };
    double upp[11][11] = {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
    };

    double lexpr_cst[11] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double uexpr_cst[11] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    size_t lexpr_dim[11][11] = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    };
    size_t uexpr_dim[11][11] = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    };

    size_t lexpr_size[11] = {11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11};

    size_t predecessor[1] = {-1};



    /* build abstract element of input layer from: inf, sup, lpp, upp, ldim, udim, lcst, ucst */
    element = fppoly_from_network_input_poly(man, 0, 11, inf, sup,
                                             lpp, lexpr_cst, lexpr_dim, upp,
                                             uexpr_cst, uexpr_dim, lexpr_size[0]);
    ffn_handle_first_relu_layer_(man, element, weights_l1_ptr, bias_l1, expr_dim_l1_ptr, 2, 11, predecessor);


    ffn_handle_intermediate_relu_layer_(man, element, weights_l2_ptr, bias_l2, expr_dim_l2_ptr, 2, 11, predecessor_l1, true);

    ffn_handle_intermediate_relu_layer_(man, element, weights_l3_ptr, bias_l3, expr_dim_l3_ptr, 2, 11, predecessor_l2, true);

    ffn_handle_last_relu_layer_(man, element, weights_out_ptr, bias_out, expr_dim_op, 2, 11, predecessor_out, false, true);

    //get bounds
    elina_linexpr0_t * lexpr = get_lexpr_for_output_neuron(man, element, 0);
    elina_linexpr0_t * uexpr = get_uexpr_for_output_neuron(man, element, 0);

    if (lexpr->size != uexpr->size) {
        lexpr = get_lexpr_for_output_neuron(man, element, 1);
        uexpr = get_uexpr_for_output_neuron(man, element, 1);
    }
    //compare class
    bool result = false;
    result = is_greater_(man, element, 1, 0, (2 + 9), true);

    if (result == false) {
        result = is_greater_(man, element, 0, 1, (2 + 9), true);

        // for debug purpose only
        ffn_handle_intermediate_relu_layer_(man, element, weights_l3_ptr, bias_l3, expr_dim_l3_ptr, output_size_l1, 8,
                                            predecessor_l2, true);
    } else {
        ffn_handle_last_relu_layer_(man, element, weights_out_ptr, bias_out, expr_dim_op, 2, 11, predecessor_out, false, true);
    }
    //ffn_handle_first_relu_layer(man, element, test_weights, bias,   input_size, input_size, predecessor);


#endif

#if HAS_RNN_TANH
    /*
     *
     * evaluate expression: *(fppoly_t*)(element->value)
     */

    /*
     *  compare layer
     */

    double weights_l[2][8] = {
            {1, -1, 0, 0, 0, 0, 0, 0},
            {-1, 1, 0, 0, 0, 0, 0, 0}
    };
    double bias_l[2] = {0, 0};

    double * weights_l_ptr[2];

    for (int i = 0; i < 2; i++) {
        weights_l_ptr[i] = &(weights_l[i]);
    }

    size_t predecessor_l[1] = {4};


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
            {-0.1, 0.1, 0, 0, 0, 0, 1.0, -1.0},
            {0.1, -0.1, 0, 0, 0, 0, 1.0, 1.0}
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

    double inf[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
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
    ffn_handle_first_tanh_layer_(man, element, weights_l1_ptr, bias_l1, expr_dim_l1_ptr, 2, 8, predecessor);


    ffn_handle_intermediate_tanh_layer_(man, element, weights_l2_ptr, bias_l2, expr_dim_l2_ptr, 2, 8, predecessor_l1, true);

    ffn_handle_intermediate_tanh_layer_(man, element, weights_l3_ptr, bias_l3, expr_dim_l3_ptr, 2, 8, predecessor_l2, true);

    ffn_handle_last_tanh_layer_(man, element, weights_out_ptr, bias_out, expr_dim_op, 2, 8, predecessor_out, false, true);

    //get bounds
    expr_t * lexpr = get_lexpr_for_output_neuron_simple(man, element, 0);
    expr_t * uexpr = get_uexpr_for_output_neuron_simple(man, element, 0);

    if (lexpr->size == uexpr->size) {
        lexpr = get_lexpr_for_output_neuron_simple(man, element, 1);
        uexpr = get_uexpr_for_output_neuron_simple(man, element, 1);
    }

    if (lexpr->size == uexpr->size) {
        lexpr = get_lexpr_for_output_neuron_simple(man, element, 1);
        uexpr = get_uexpr_for_output_neuron_simple(man, element, 1);
    }

    ffn_handle_last_tanh_layer_(man, element, weights_l_ptr, bias_l, expr_dim_op, 2, 8, predecessor_l, false, true);

    expr_t * lexprY = get_lexpr_for_output_neuron_simple(man, element, 0);
    expr_t * uexprY = get_uexpr_for_output_neuron_simple(man, element, 0);

    if (lexprY->size == uexprY->size) {
        lexprY = get_lexpr_for_output_neuron_simple(man, element, 1);
        uexprY = get_uexpr_for_output_neuron_simple(man, element, 1);
    }

    double out_lb, out_last;
    out_lb =  lb_for_neuron(man, element, 4, 0);
    out_last = lb_for_neuron(man, element, 4, 1);

    if (out_lb == out_last) {
        ffn_handle_last_tanh_layer_(man, element, weights_out_ptr, bias_out, expr_dim_op, 2, 8, predecessor_out, false, true);
    }
    //compare class
    bool result = false;
    result = is_greater_(man, element, 0, 1, (2 + 6), true);
    result = is_greater_(man, element, 1, 0, (2 + 6), true);
    if (result == false) {
        result = is_greater_(man, element, 0, 1, (2 + 6), true);

        // for debug purpose only
        ffn_handle_intermediate_tanh_layer_(man, element, weights_l3_ptr, bias_l3, expr_dim_l3_ptr, 2, 8,
                                            predecessor_l2, true);
    } else {
        ffn_handle_last_tanh_layer_(man, element, weights_out_ptr, bias_out, expr_dim_op, 2, 8, predecessor_out, false, true);
    }

#endif
#if HAS_RNN_EXAMPLE
    /*
     *
     * evaluate expression: *(fppoly_t*)(element->value)
     */

    /*
     *  compare layer
     */

    double weights_l[2][8] = {
            {1, -1, 0, 0, 0, 0, 0, 0},
            {-1, 1, 0, 0, 0, 0, 0, 0}
    };
    double bias_l[2] = {0, 0};

    double * weights_l_ptr[2];

    for (int i = 0; i < 2; i++) {
        weights_l_ptr[i] = &(weights_l[i]);
    }

    size_t predecessor_l[1] = {4};


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

    double inf[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
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

    //get bounds
    expr_t * lexpr = get_lexpr_for_output_neuron_simple(man, element, 0);
    expr_t * uexpr = get_uexpr_for_output_neuron_simple(man, element, 0);

    if (lexpr->size == uexpr->size) {
        lexpr = get_lexpr_for_output_neuron_simple(man, element, 1);
        uexpr = get_uexpr_for_output_neuron_simple(man, element, 1);
    }

    if (lexpr->size == uexpr->size) {
        lexpr = get_lexpr_for_output_neuron_simple(man, element, 1);
        uexpr = get_uexpr_for_output_neuron_simple(man, element, 1);
    }

    ffn_handle_last_relu_layer_(man, element, weights_l_ptr, bias_l, expr_dim_op, 2, 8, predecessor_l, false, true);

    expr_t * lexprY = get_lexpr_for_output_neuron_simple(man, element, 0);
    expr_t * uexprY = get_uexpr_for_output_neuron_simple(man, element, 0);

    if (lexprY->size == uexprY->size) {
        lexprY = get_lexpr_for_output_neuron_simple(man, element, 1);
        uexprY = get_uexpr_for_output_neuron_simple(man, element, 1);
    }

    double out_lb, out_last;
    out_lb =  lb_for_neuron(man, element, 4, 0);
    out_last = lb_for_neuron(man, element, 4, 1);

    if (out_lb == out_last) {
        ffn_handle_last_relu_layer_(man, element, weights_out_ptr, bias_out, expr_dim_op, 2, 8, predecessor_out, false, true);
    }
    //compare class
    bool result = false;
    result = is_greater_(man, element, 0, 1, (2 + 6), true);
    result = is_greater_(man, element, 1, 0, (2 + 6), true);
    if (result == false) {
        result = is_greater_(man, element, 0, 1, (2 + 6), true);

        // for debug purpose only
        ffn_handle_intermediate_relu_layer_(man, element, weights_l3_ptr, bias_l3, expr_dim_l3_ptr, 2, 8,
                                            predecessor_l2, true);
    } else {
        ffn_handle_last_relu_layer_(man, element, weights_out_ptr, bias_out, expr_dim_op, 2, 8, predecessor_out, false, true);
    }
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