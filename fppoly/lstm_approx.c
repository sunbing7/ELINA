#include "lstm_approx.h"


#if HAS_LSTM
expr_t * lexpr_replace_maxpool_or_lstm_bounds_(fppoly_internal_t * pr, expr_t * expr, neuron_t ** neurons, size_t num_neu_out){
    //printf("begin\n");
    //fflush(stdout);
    size_t out_num_neurons = num_neu_out;  // how many neurons in previous layer
    size_t num_neurons = expr->size;
    size_t i,k;
    expr_t *res;
    if(expr->type==DENSE){
        k = 0;
    }
    else{
        k = expr->dim[0];
    }

    neuron_t *neuron_k = neurons[k];
    if(expr->sup_coeff[0]<0){
        //expr_print(neuron_k->uexpr);
        if(neuron_k->uexpr==NULL){
            res = (expr_t *)malloc(sizeof(expr_t));
            if(expr->size > 0){
                res->inf_coeff = malloc(expr->size*sizeof(double));
                res->sup_coeff = malloc(expr->size*sizeof(double));
                memset( res->inf_coeff, 0, expr->size*sizeof(double));
                memset( res->sup_coeff, 0, expr->size*sizeof(double));
            } else {
                res->inf_coeff = res->sup_coeff = NULL;
            }
            res->dim = NULL;
            res->size = 0;
            res->type = SPARSE;
            elina_double_interval_mul_cst_coeff(pr,&res->inf_cst,&res->sup_cst,neuron_k->lb,neuron_k->ub,expr->inf_coeff[0],expr->sup_coeff[0]);
        }
        else{
            res = multiply_expr(pr,neuron_k->uexpr,expr->inf_coeff[0],expr->sup_coeff[0]);
        }
        //printf("multiply end %zu \n",k);
        //expr_print(res);
        //fflush(stdout);
    }
    else if(expr->inf_coeff[0]<0){
        //expr_print(neuron_k->lexpr);
        if(neuron_k->lexpr==NULL){
            res = (expr_t *)malloc(sizeof(expr_t));
            if(expr->size > 0){
                res->inf_coeff = malloc(expr->size*sizeof(double));
                res->sup_coeff = malloc(expr->size*sizeof(double));
                memset( res->inf_coeff, 0, expr->size*sizeof(double));
                memset( res->sup_coeff, 0, expr->size*sizeof(double));
            } else {
                res->inf_coeff = res->sup_coeff = NULL;
            }
            res->dim = NULL;
            res->size = 0;
            res->type = SPARSE;
            elina_double_interval_mul_cst_coeff(pr,&res->inf_cst,&res->sup_cst,neuron_k->lb,neuron_k->ub,expr->inf_coeff[0],expr->sup_coeff[0]);
        }
        else{
            res = multiply_expr(pr,neuron_k->lexpr,expr->inf_coeff[0],expr->sup_coeff[0]);
        }
        //printf("multiply end %zu \n",k);
        //expr_print(res);
        //fflush(stdout);
    }
    else{
        //printf("WTF1\n");
        //fflush(stdout);
        double tmp1, tmp2;
        elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,neuron_k->lb,neuron_k->ub,expr->inf_coeff[0],expr->sup_coeff[0]);
        double coeff[1];
        size_t dim[1];
        coeff[0] = 0;
        dim[0] = 0;
        res = create_sparse_expr(coeff,-tmp1,dim,1);
    }

    //printf("middle\n");
    //fflush(stdout);
    expr_t * mul_expr;
    for(i = 1; i < num_neurons; i++){
        if(expr->type==DENSE){
            k = i;
        }
        else{
            k = expr->dim[i];
        }

#if HAS_LSTM
        if (expr->type == SPARSE){
            if ((k + 1) > out_num_neurons) {
                if (res->size > k) {    /* !!!!sunbing  */
                    res->inf_coeff[k] = res->inf_coeff[k] + expr->inf_coeff[k];
                    res->sup_coeff[k] = res->sup_coeff[k] + expr->sup_coeff[k];
                }
                continue;
            }
        }
#endif
        neuron_t *neuron_k = neurons[k];
        if(expr->sup_coeff[i]<0){
            //expr_print(neuron_k->uexpr);
            //printf("add start %zu %zu\n",k,i);

            //expr_print(res);

            if(neuron_k->uexpr==NULL){
                mul_expr = (expr_t *)malloc(sizeof(expr_t));
                mul_expr->inf_coeff = mul_expr->sup_coeff = NULL;
                mul_expr->dim = NULL;
                mul_expr->size = 0;
                mul_expr->type = SPARSE;
                //printf("lb: %g %g\n");
                elina_double_interval_mul_cst_coeff(pr,&mul_expr->inf_cst,&mul_expr->sup_cst,neuron_k->lb,neuron_k->ub,expr->inf_coeff[i],expr->sup_coeff[i]);
                res->inf_cst += mul_expr->inf_cst;
                res->sup_cst += mul_expr->sup_cst;
            } else{
                mul_expr = multiply_expr(pr,neuron_k->uexpr,expr->inf_coeff[i],expr->sup_coeff[i]);
                add_expr(pr,res,mul_expr);

            }
            //expr_print(mul_expr);
            //fflush(stdout);
            //printf("add finish\n");
            //expr_print(res);
            //fflush(stdout);
            free_expr(mul_expr);
        } else if (expr->inf_coeff[i]<0) {
            //expr_print(neuron_k->lexpr);
            //printf("add start %zu %zu\n",k,i);

            //expr_print(res);

            if(neuron_k->lexpr==NULL){
                mul_expr = (expr_t *)malloc(sizeof(expr_t));
                mul_expr->inf_coeff = mul_expr->sup_coeff = NULL;
                mul_expr->dim = NULL;
                mul_expr->size = 0;
                mul_expr->type = SPARSE;
                elina_double_interval_mul_cst_coeff(pr,&mul_expr->inf_cst,&mul_expr->sup_cst,neuron_k->lb,neuron_k->ub,expr->inf_coeff[i],expr->sup_coeff[i]);
                res->inf_cst += mul_expr->inf_cst;
                res->sup_cst += mul_expr->sup_cst;
            } else {
                mul_expr = multiply_expr(pr,neuron_k->lexpr,expr->inf_coeff[i],expr->sup_coeff[i]);
                //printf("add start1 %zu %zu\n",k,i);
                //expr_print(res);
                //expr_print(mul_expr);
                //fflush(stdout);
                add_expr(pr,res,mul_expr);

            }
            //expr_print(mul_expr);
            //	fflush(stdout);
            //printf("add finish1\n");
            //expr_print(res);
            //fflush(stdout);
            free_expr(mul_expr);
        } else {
            //printf("WTF2\n");
            //fflush(stdout);
            double tmp1, tmp2;
            elina_double_interval_mul_expr_coeff(pr,&tmp1,&tmp2,neuron_k->lb,neuron_k->ub,expr->inf_coeff[i],expr->sup_coeff[i]);
            res->inf_cst = res->inf_cst + tmp1;
            res->sup_cst = res->sup_cst - tmp1;
        }
    }
    //printf("finish\n");
    //fflush(stdout);
    res->inf_cst = res->inf_cst + expr->inf_cst;
    res->sup_cst = res->sup_cst + expr->sup_cst;

    return res;
}

expr_t * uexpr_replace_maxpool_or_lstm_bounds_(fppoly_internal_t * pr, expr_t * expr, neuron_t ** neurons, size_t num_neu_out){
    size_t out_num_neurons = num_neu_out;  // how many neurons in previous layer
    size_t num_neurons = expr->size;
    size_t i,k;
    expr_t *res;
    if(expr->type==DENSE){
        k = 0;
    }
    else{
        k = expr->dim[0];

        //check before everything
#if 0
        for (int idx = 0; idx < out_num_neurons; idx++) {
            for (int dim_idx = 0; dim_idx < neurons[idx]->uexpr->size; dim_idx++ ) {
                if (neurons[idx]->uexpr->dim[dim_idx] != dim_idx)
                    printf("Wrong value!");
            }
        }
#endif
    }

    neuron_t *neuron_k = neurons[k];
    if(expr->sup_coeff[0]<0){
        res = multiply_expr(pr,neuron_k->lexpr,expr->inf_coeff[0],expr->sup_coeff[0]);
    }
    else if(expr->inf_coeff[0]<0){
        res = multiply_expr(pr,neuron_k->uexpr,expr->inf_coeff[0],expr->sup_coeff[0]);
    }
    else{

        double tmp1, tmp2;
        elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,neuron_k->lb,neuron_k->ub,expr->inf_coeff[0],expr->sup_coeff[0]);
        double coeff[1];
        size_t dim[1];
        coeff[0] = 0;
        dim[0] = 0;
        res = create_sparse_expr(coeff,tmp2,dim,1);
    }

    //check before everything
#if 0
    if (expr->type == SPARSE)
        for (int idx = 0; idx < out_num_neurons; idx++) {
                for (int x = 0; x < neurons[idx]->uexpr->size; x++) {
                    if (neurons[idx]->uexpr->dim[x] != x)
                        printf("Wrong value!");
                }
        }
#endif

    for(i = 1; i < num_neurons; i++){
        if(expr->type==DENSE){
            k = i;
        }
        else{
            k = expr->dim[i];

                //for (int idx = 0; idx < out_num_neurons; idx++) {
                //    for (int x = 0; x < neurons[idx]->uexpr->size; x++) {
                //        if (neurons[idx]->uexpr->dim[x] != x)
                //            printf("Wrong value!");
                //    }
                //}
        }

#if HAS_LSTM
        if (expr->type == SPARSE){

            if ((k + 1) > out_num_neurons) {
                if (res->size > k) {    /* !!!!sunbing  */
                    res->inf_coeff[k] = res->inf_coeff[k] + expr->inf_coeff[k];
                    res->sup_coeff[k] = res->sup_coeff[k] + expr->sup_coeff[k];
                }
                continue;
            }
        }
#endif
        neuron_t *neuron_k = neurons[k];
        if(expr->sup_coeff[i]<0){
            expr_t * mul_expr = multiply_expr(pr,neuron_k->lexpr,expr->inf_coeff[i],expr->sup_coeff[i]);
            add_expr(pr,res,mul_expr);
            free_expr(mul_expr);
        }
        else if (expr->inf_coeff[i]<0){

            expr_t * mul_expr = multiply_expr(pr,neuron_k->uexpr,expr->inf_coeff[i],expr->sup_coeff[i]);

            add_expr(pr,res,mul_expr);

            free_expr(mul_expr);
        }
        else{

            double tmp1, tmp2;
            elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,neuron_k->lb,neuron_k->ub,expr->inf_coeff[i],expr->sup_coeff[i]);
            res->inf_cst = res->inf_cst - tmp2;
            res->sup_cst = res->sup_cst + tmp2;
        }

    }
    res->inf_cst = res->inf_cst + expr->inf_cst;
    res->sup_cst = res->sup_cst + expr->sup_cst;
    return res;
}

#endif
expr_t * lexpr_replace_maxpool_or_lstm_bounds(fppoly_internal_t * pr, expr_t * expr, neuron_t ** neurons){
	//printf("begin\n");
	//fflush(stdout);

	size_t num_neurons = expr->size;
	size_t i,k;
	expr_t *res;
	if(expr->type==DENSE){
		k = 0;
	}
	else{
		k = expr->dim[0];
	}
	neuron_t *neuron_k = neurons[k];
	if(expr->sup_coeff[0]<0){
		//expr_print(neuron_k->uexpr);
		if(neuron_k->uexpr==NULL){
			res = (expr_t *)malloc(sizeof(expr_t));
			res->inf_coeff = res->sup_coeff =  NULL;
			res->dim = NULL;
			res->size = 0;
			res->type = SPARSE;
			elina_double_interval_mul_cst_coeff(pr,&res->inf_cst,&res->sup_cst,neuron_k->lb,neuron_k->ub,expr->inf_coeff[0],expr->sup_coeff[0]);
		}
		else{
			res = multiply_expr(pr,neuron_k->uexpr,expr->inf_coeff[0],expr->sup_coeff[0]);
		}
		//printf("multiply end %zu \n",k);
		//expr_print(res);
		//fflush(stdout);
	}
	else if(expr->inf_coeff[0]<0){
		//expr_print(neuron_k->lexpr);
		if(neuron_k->lexpr==NULL){
			res = (expr_t *)malloc(sizeof(expr_t));
			res->inf_coeff = res->sup_coeff = NULL;
			res->dim = NULL;
			res->size = 0;
			res->type = SPARSE;
			elina_double_interval_mul_cst_coeff(pr,&res->inf_cst,&res->sup_cst,neuron_k->lb,neuron_k->ub,expr->inf_coeff[0],expr->sup_coeff[0]);
		}
		else{
			res = multiply_expr(pr,neuron_k->lexpr,expr->inf_coeff[0],expr->sup_coeff[0]);
		}
		//printf("multiply end %zu \n",k);
		//expr_print(res);
		//fflush(stdout);
	}
	else{
		//printf("WTF1\n");
		//fflush(stdout);
		double tmp1, tmp2;
		elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,neuron_k->lb,neuron_k->ub,expr->inf_coeff[0],expr->sup_coeff[0]);
		double coeff[1];
		size_t dim[1];
		coeff[0] = 0;
		dim[0] = 0;
		res = create_sparse_expr(coeff,-tmp1,dim,1);
	}
	//printf("middle\n");
	//fflush(stdout);
	expr_t * mul_expr;
	for(i = 1; i < num_neurons; i++){
		if(expr->type==DENSE){
			k = i;
		}
		else{
			k = expr->dim[i];
		}

		neuron_t *neuron_k = neurons[k];
		if(expr->sup_coeff[i]<0){
			//expr_print(neuron_k->uexpr);
			//printf("add start %zu %zu\n",k,i);
			
				//expr_print(res);
				
			if(neuron_k->uexpr==NULL){
				mul_expr = (expr_t *)malloc(sizeof(expr_t));
				mul_expr->inf_coeff = mul_expr->sup_coeff = NULL;
				mul_expr->dim = NULL;
				mul_expr->size = 0;
				mul_expr->type = SPARSE;
				//printf("lb: %g %g\n");
				elina_double_interval_mul_cst_coeff(pr,&mul_expr->inf_cst,&mul_expr->sup_cst,neuron_k->lb,neuron_k->ub,expr->inf_coeff[i],expr->sup_coeff[i]);
				res->inf_cst += mul_expr->inf_cst;
				res->sup_cst += mul_expr->sup_cst;
			}
			else{
				mul_expr = multiply_expr(pr,neuron_k->uexpr,expr->inf_coeff[i],expr->sup_coeff[i]);
				
				add_expr(pr,res,mul_expr);
				
			}
			//expr_print(mul_expr);
				//fflush(stdout);
			//printf("add finish\n");
				//expr_print(res);
				//fflush(stdout);
			free_expr(mul_expr);
		}
		else if (expr->inf_coeff[i]<0){
			//expr_print(neuron_k->lexpr);
			//printf("add start %zu %zu\n",k,i);
			
				//expr_print(res);
				
			if(neuron_k->lexpr==NULL){
				mul_expr = (expr_t *)malloc(sizeof(expr_t));
				mul_expr->inf_coeff = mul_expr->sup_coeff = NULL;
				mul_expr->dim = NULL;
				mul_expr->size = 0;
				mul_expr->type = SPARSE;
				elina_double_interval_mul_cst_coeff(pr,&mul_expr->inf_cst,&mul_expr->sup_cst,neuron_k->lb,neuron_k->ub,expr->inf_coeff[i],expr->sup_coeff[i]);
				res->inf_cst += mul_expr->inf_cst;
				res->sup_cst += mul_expr->sup_cst;
			}
			else{
				mul_expr = multiply_expr(pr,neuron_k->lexpr,expr->inf_coeff[i],expr->sup_coeff[i]);
				//printf("add start1 %zu %zu\n",k,i);
				//expr_print(res);
				//expr_print(mul_expr);
				//fflush(stdout);
				add_expr(pr,res,mul_expr);
				
			}
			//expr_print(mul_expr);
			//	fflush(stdout);
			//printf("add finish1\n");
			//expr_print(res);
			//fflush(stdout);
			free_expr(mul_expr);
		}
		else{
			//printf("WTF2\n");
			//fflush(stdout);
			double tmp1, tmp2;
			elina_double_interval_mul_expr_coeff(pr,&tmp1,&tmp2,neuron_k->lb,neuron_k->ub,expr->inf_coeff[i],expr->sup_coeff[i]);
			res->inf_cst = res->inf_cst + tmp1;
			res->sup_cst = res->sup_cst - tmp1;
		}
	}
	//printf("finish\n");	
	//fflush(stdout);
	res->inf_cst = res->inf_cst + expr->inf_cst;
	res->sup_cst = res->sup_cst + expr->sup_cst;
	return res;
}

expr_t * uexpr_replace_maxpool_or_lstm_bounds(fppoly_internal_t * pr, expr_t * expr, neuron_t ** neurons){
	size_t num_neurons = expr->size;
	size_t i,k;
	expr_t *res;
	if(expr->type==DENSE){
		k = 0;
	}
	else{
		k = expr->dim[0];
	}
	neuron_t *neuron_k = neurons[k];
	if(expr->sup_coeff[0]<0){
		res = multiply_expr(pr,neuron_k->lexpr,expr->inf_coeff[0],expr->sup_coeff[0]);
	}
	else if(expr->inf_coeff[0]<0){
		res = multiply_expr(pr,neuron_k->uexpr,expr->inf_coeff[0],expr->sup_coeff[0]);
	}
	else{
		
		double tmp1, tmp2;
		elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,neuron_k->lb,neuron_k->ub,expr->inf_coeff[0],expr->sup_coeff[0]);
		double coeff[1];
		size_t dim[1];
		coeff[0] = 0;
		dim[0] = 0;
		res = create_sparse_expr(coeff,tmp2,dim,1);
	}

	for(i = 1; i < num_neurons; i++){
		if(expr->type==DENSE){
			k = i;
		}
		else{
			k = expr->dim[i];
		}
		neuron_t *neuron_k = neurons[k];
		if(expr->sup_coeff[i]<0){
			expr_t * mul_expr = multiply_expr(pr,neuron_k->lexpr,expr->inf_coeff[i],expr->sup_coeff[i]);
			add_expr(pr,res,mul_expr);
			free_expr(mul_expr);
		}
		else if (expr->inf_coeff[i]<0){
			expr_t * mul_expr = multiply_expr(pr,neuron_k->uexpr,expr->inf_coeff[i],expr->sup_coeff[i]);
			add_expr(pr,res,mul_expr);
			free_expr(mul_expr);
		}
		else{
			
			double tmp1, tmp2;
			elina_double_interval_mul_cst_coeff(pr,&tmp1,&tmp2,neuron_k->lb,neuron_k->ub,expr->inf_coeff[i],expr->sup_coeff[i]);
			res->inf_cst = res->inf_cst - tmp2;
			res->sup_cst = res->sup_cst + tmp2;
		}
	}
	res->inf_cst = res->inf_cst + expr->inf_cst;
	res->sup_cst = res->sup_cst + expr->sup_cst;
	return res;
}

expr_t * lexpr_unroll_lstm_layer(fppoly_internal_t *pr, expr_t * expr, neuron_t ** neurons){
	return NULL;
}


void create_lstm_layer(elina_manager_t *man, elina_abstract0_t *abs, size_t h, size_t *predecessors){
	fppoly_t *fp = fppoly_of_abstract0(abs);
	size_t numlayers = fp->numlayers;

	fppoly_add_new_layer(fp,h, LSTM, NONE);
	fp->lstm_index = numlayers;
}

void create_first_lstm_layer(elina_manager_t *man, elina_abstract0_t *abs, size_t h, size_t *predecessors){
    fppoly_t *fp = fppoly_of_abstract0(abs);
    size_t numlayers = fp->numlayers;

    fppoly_alloc_first_layer(fp,h, LSTM, NONE);
    fp->lstm_index = numlayers;
}

void handle_lstm_layer_(elina_manager_t *man, elina_abstract0_t *abs, double **weights,  double *bias, size_t d, size_t h, size_t * dim, size_t * predecessors, bool use_area_heuristic){
	fppoly_t *fp = fppoly_of_abstract0(abs);
	fppoly_internal_t *pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	size_t lstm_index = fp->lstm_index;
	layer_t *layer = fp->layers[lstm_index];
	neuron_t **out_neurons = fp->layers[lstm_index]->neurons;
	fp->layers[lstm_index]->predecessors = predecessors;
	size_t i;
	neuron_t * neuron = neuron_alloc();
	bool first_time_step = (layer->h_t_inf==NULL && layer->h_t_sup==NULL);
	size_t k = h + d;
	if(first_time_step){

	    layer->h_t_inf = (double*)malloc(h*sizeof(double));
		layer->h_t_sup = (double*)malloc(h*sizeof(double));
		layer->c_t_inf = (double*)malloc(h*sizeof(double));
		layer->c_t_sup = (double*)malloc(h*sizeof(double));
	}

	for(i=0; i< 1; i++){
	  //printf("i = %d\n",(int)i);
		expr_t *f_t_lexpr, *i_t_lexpr, *o_t_lexpr, *c_t_lexpr;
		if(first_time_step){
#if HAS_LSTM
            i_t_lexpr =  create_sparse_expr(weights[i],bias[i], dim, d);
            c_t_lexpr =  create_sparse_expr(weights[h+i],bias[h+i],dim, d);
            f_t_lexpr =  create_sparse_expr(weights[2*h+i],bias[2*h+i], dim, d);
            o_t_lexpr =  create_sparse_expr(weights[3*h+i],bias[3*h+i], dim, d);
#else
		    i_t_lexpr =  create_dense_expr(weights[i],bias[i],d);
			c_t_lexpr =  create_dense_expr(weights[h+i],bias[h+i],d);	
			f_t_lexpr =  create_dense_expr(weights[2*h+i],bias[2*h+i],d);
			o_t_lexpr =  create_dense_expr(weights[3*h+i],bias[3*h+i],d);
#endif
		}		
		else{
#if HAS_LSTM
            expr_t * tmp1 = create_sparse_expr(weights[i],bias[i], dim, d);
            expr_t * tmp2 = create_sparse_expr(weights[h+i],bias[h+i],dim, d);
            expr_t * tmp3 = create_sparse_expr(weights[2*h+i],bias[2*h+i], dim, d);
            expr_t * tmp4 = create_sparse_expr(weights[3*h+i],bias[3*h+i], dim, d);
            //todo
            //i_t_lexpr = concretize_dense_sub_expr(pr, tmp1, layer->h_t_inf, layer->h_t_sup, d, d+h);
            //c_t_lexpr = concretize_dense_sub_expr(pr, tmp2, layer->h_t_inf, layer->h_t_sup, d, d+h);
            //f_t_lexpr = concretize_dense_sub_expr(pr, tmp3, layer->h_t_inf, layer->h_t_sup, d, d+h);
            //o_t_lexpr = concretize_dense_sub_expr(pr, tmp4, layer->h_t_inf, layer->h_t_sup, d, d+h);
#else
		    expr_t * tmp1 = create_dense_expr(weights[i],bias[i],d+h);
			expr_t * tmp2 = create_dense_expr(weights[h+i],bias[h+i],d+h);
			expr_t * tmp3 = create_dense_expr(weights[2*h+i],bias[2*h+i],d+h);
			expr_t * tmp4 = create_dense_expr(weights[3*h+i],bias[3*h+i],d+h);
			i_t_lexpr = concretize_dense_sub_expr(pr, tmp1, layer->h_t_inf, layer->h_t_sup, d, d+h);
			c_t_lexpr = concretize_dense_sub_expr(pr, tmp2, layer->h_t_inf, layer->h_t_sup, d, d+h);
			f_t_lexpr = concretize_dense_sub_expr(pr, tmp3, layer->h_t_inf, layer->h_t_sup, d, d+h);
			o_t_lexpr = concretize_dense_sub_expr(pr, tmp4, layer->h_t_inf, layer->h_t_sup, d, d+h);
#endif
			free_expr(tmp1);	
			free_expr(tmp2);
			free_expr(tmp3);
			free_expr(tmp4);
		}

		//expr_print(f_t_lexpr);

		//printf("computing forget...\n");
		expr_t *f_t_uexpr = copy_expr(f_t_lexpr);
		expr_t *tmp_f_t_lexpr = copy_expr(f_t_lexpr);
		expr_t *tmp_f_t_uexpr = copy_expr(f_t_uexpr);
		double lb_f_t = get_lb_using_previous_layers(man, fp, tmp_f_t_lexpr, lstm_index,use_area_heuristic);
		double ub_f_t = get_ub_using_previous_layers(man, fp, tmp_f_t_uexpr, lstm_index,use_area_heuristic);
		/* free_expr(tmp_f_t_lexpr); */
		/* free_expr(tmp_f_t_uexpr); */

		neuron->lb = lb_f_t;
		neuron->ub = ub_f_t;
		//printf("forget gate before sigmoid: lb = %lf, ub = %lf\n",neuron->lb, neuron->ub);
		//expr_print(f_t_lexpr);
		//expr_print(f_t_uexpr);
		lb_f_t = apply_sigmoid_lexpr(pr, &f_t_lexpr, neuron);
		ub_f_t = apply_sigmoid_uexpr(pr, &f_t_uexpr, neuron);
		//printf("forget gate after sigmoid: lb_f_t = %lf, ub_f_t = %lf\n",lb_f_t,ub_f_t);
		//expr_print(f_t_lexpr);
		//expr_print(f_t_uexpr);
		//printf("forget gate done\n\n");

		//printf("computing input...\n");
		expr_t *i_t_uexpr = copy_expr(i_t_lexpr);
		expr_t *tmp_i_t_lexpr = copy_expr(i_t_lexpr);
		expr_t *tmp_i_t_uexpr = copy_expr(i_t_uexpr);
		double lb_i_t = get_lb_using_previous_layers(man, fp, tmp_i_t_lexpr,lstm_index,use_area_heuristic);
		double ub_i_t = get_ub_using_previous_layers(man, fp, tmp_i_t_uexpr, lstm_index,use_area_heuristic);	
		/* free_expr(tmp_i_t_lexpr); */
		/* free_expr(tmp_i_t_uexpr); */
		neuron->lb = lb_i_t;
		neuron->ub = ub_i_t;
		//printf("input gate before sigmoid: lb = %lf, ub = %lf\n",neuron->lb, neuron->ub);
		//expr_print(i_t_uexpr);
		lb_i_t = apply_sigmoid_lexpr(pr, &i_t_lexpr, neuron);
		ub_i_t = apply_sigmoid_uexpr(pr, &i_t_uexpr, neuron);
		//expr_print(i_t_uexpr);
		//printf("input gate after sigmoid: lb_i_t = %lf, ub_i_t = %lf\n",lb_i_t,ub_i_t);
		//printf("input gate done\n\n");

		//printf("computing output...\n");
		expr_t *o_t_uexpr = copy_expr(o_t_lexpr);
		expr_t *tmp_o_t_lexpr = copy_expr(o_t_lexpr);
		expr_t *tmp_o_t_uexpr = copy_expr(o_t_uexpr);
		double lb_o_t = get_lb_using_previous_layers(man, fp, tmp_o_t_lexpr, lstm_index,use_area_heuristic);
		double ub_o_t = get_ub_using_previous_layers(man, fp, tmp_o_t_uexpr, lstm_index,use_area_heuristic);
		/* free_expr(tmp_o_t_lexpr); */
		/* free_expr(tmp_o_t_uexpr); */

		neuron->lb = lb_o_t;
		neuron->ub = ub_o_t;		
		//printf("output gate before sigmoid: lb = %lf, ub = %lf\n",neuron->lb, neuron->ub);
		lb_o_t = apply_sigmoid_lexpr(pr, &o_t_lexpr, neuron);
		ub_o_t = apply_sigmoid_uexpr(pr, &o_t_uexpr, neuron);
		//printf("output gate after sigmoid: lb = %lf, ub = %lf\n",lb_o_t,ub_o_t);
		out_neurons[i]->lb = lb_o_t;
		out_neurons[i]->ub = ub_o_t;
		out_neurons[i]->lexpr = o_t_lexpr;
		out_neurons[i]->uexpr = o_t_uexpr;
		//printf("output gate done\n\n");

		//printf("computing control state...\n");
		//printf("control expression:\n");
		//expr_print(c_t_lexpr);
		//printf("...\n");
		expr_t *c_t_uexpr = copy_expr(c_t_lexpr);
		expr_t *tmp_c_t_lexpr = copy_expr(c_t_lexpr);
		expr_t *tmp_c_t_uexpr = copy_expr(c_t_uexpr);
		double lb_c_t = get_lb_using_previous_layers(man, fp, tmp_c_t_lexpr, lstm_index,use_area_heuristic);
		double ub_c_t = get_ub_using_previous_layers(man, fp, tmp_c_t_uexpr, lstm_index,use_area_heuristic);
		neuron->lb = lb_c_t;
		neuron->ub = ub_c_t;
		//expr_print(c_t_lexpr);
		//expr_print(c_t_uexpr);
		//printf("control before tanh: lb = %lf, ub = %lf\n",neuron->lb,neuron->ub);
		lb_c_t = apply_tanh_lexpr(pr,&c_t_lexpr, neuron);
		ub_c_t = apply_tanh_uexpr(pr,&c_t_uexpr, neuron);			
		//printf("control after tanh: lb = %lf, ub = %lf\n",lb_c_t,ub_c_t);
		//printf("control expression:\n");
		//expr_print(c_t_lexpr);
		//expr_print(c_t_uexpr);

		//printf("=======================\n");

		//printf("multiplying control by input:\n");
		expr_t *tmp_l, *tmp_u;
		double width1 = ub_i_t + lb_i_t;
		double width2 = ub_c_t + lb_c_t;
		tmp_l = c_t_lexpr;
		tmp_u = c_t_uexpr;
		//printf("control: [%lf %lf], input: [%lf %lf]\n",lb_c_t,ub_c_t,lb_i_t,ub_i_t);
		//printf("control before multiplying by input:\n");
		//expr_print(c_t_lexpr);
		//expr_print(c_t_uexpr);
		if(width1 < width2){
		  //printf("concretize input\n");
			c_t_lexpr = multiply_expr(pr,c_t_lexpr,lb_i_t,ub_i_t);
			c_t_uexpr = multiply_expr(pr,c_t_uexpr,lb_i_t,ub_i_t);
		}
		else{
		  //printf("concretize control\n");
            if(lb_c_t<0){
                c_t_lexpr = multiply_expr(pr,i_t_lexpr,lb_c_t,ub_c_t);
                c_t_uexpr = multiply_expr(pr,i_t_uexpr,lb_c_t,ub_c_t);
            }
            else if(ub_c_t<0){
                c_t_lexpr = multiply_expr(pr,i_t_uexpr,lb_c_t,ub_c_t);
                c_t_uexpr = multiply_expr(pr,i_t_lexpr,lb_c_t,ub_c_t);
            }
            else{
                c_t_lexpr = multiply_expr(pr,i_t_lexpr,0,0);
                c_t_uexpr = multiply_expr(pr,i_t_uexpr,0,0);
                double tmp1, tmp2;
                elina_double_interval_mul_expr_coeff(pr,&tmp1,&tmp2,lb_i_t,ub_i_t,lb_c_t,ub_c_t);
                c_t_lexpr->inf_cst += tmp1;
                c_t_lexpr->sup_cst += tmp2;
                c_t_uexpr->inf_cst += tmp1;
                c_t_uexpr->sup_cst += tmp2;
            }
		}

		//printf("control after multiplying by input:\n");
		//expr_print(c_t_lexpr);
		//expr_print(c_t_uexpr);

		free_expr(tmp_l);
		free_expr(tmp_u);

		//printf("here\n\n\n");
		//printf("====================================\n");
		
		if(!first_time_step){
            if(layer->c_t_inf[i]<0){
                tmp_l = multiply_expr(pr,f_t_lexpr,layer->c_t_inf[i],layer->c_t_sup[i]);
                tmp_u = multiply_expr(pr,f_t_uexpr,layer->c_t_inf[i],layer->c_t_sup[i]);
            }
            else if(layer->c_t_sup[i]<0){
                tmp_l = multiply_expr(pr,f_t_uexpr,layer->c_t_inf[i],layer->c_t_sup[i]);
                tmp_u = multiply_expr(pr,f_t_lexpr,layer->c_t_inf[i],layer->c_t_sup[i]);
            }
            else{
                tmp_l = multiply_expr(pr,f_t_lexpr,0,0);
                tmp_u = multiply_expr(pr,f_t_uexpr,0,0);
                double tmp1, tmp2;
                elina_double_interval_mul_expr_coeff(pr,&tmp1,&tmp2,lb_f_t,ub_f_t,layer->c_t_inf[i],layer->c_t_sup[i]);
                tmp_l->inf_cst += tmp1;
                tmp_l->sup_cst += tmp2;
                tmp_u->inf_cst += tmp1;
                tmp_u->sup_cst += tmp2;
            }
			add_expr(pr,c_t_lexpr,tmp_l);
			add_expr(pr,c_t_uexpr,tmp_u);
			free_expr(tmp_l);
			free_expr(tmp_u);
		}
		layer->c_t_inf[i] = get_lb_using_previous_layers(man, fp, c_t_lexpr, lstm_index,use_area_heuristic);
		layer->c_t_sup[i] = get_ub_using_previous_layers(man, fp, c_t_uexpr, lstm_index,use_area_heuristic);

		neuron->lb = layer->c_t_inf[i];
		neuron->ub = layer->c_t_sup[i];

		//printf("c_t ---> lb = %lf, ub = %lf\n", neuron->lb, neuron->ub);

		lb_c_t = apply_tanh_lexpr(pr,&c_t_lexpr, neuron);
		ub_c_t = apply_tanh_uexpr(pr,&c_t_uexpr, neuron);
		
		width1 = ub_o_t + lb_o_t;
		width2 = ub_c_t + lb_c_t; 

		expr_t * h_t_lexpr, *h_t_uexpr;
		if(width1<width2){
			h_t_lexpr = multiply_expr(pr,c_t_lexpr,lb_o_t,ub_o_t);
			h_t_uexpr = multiply_expr(pr,c_t_uexpr,lb_o_t,ub_o_t);
		}
		else{
			h_t_lexpr =  multiply_expr(pr,o_t_lexpr,lb_c_t,ub_c_t);
			h_t_uexpr =  multiply_expr(pr,o_t_uexpr,lb_c_t,ub_c_t);
		}

		layer->h_t_inf[i] = get_lb_using_previous_layers(man, fp, h_t_lexpr, lstm_index,use_area_heuristic);
		layer->h_t_sup[i] = get_ub_using_previous_layers(man, fp, h_t_uexpr, lstm_index,use_area_heuristic);

		free_expr(f_t_lexpr);
		free_expr(f_t_uexpr);
		free_expr(i_t_lexpr);
		free_expr(i_t_uexpr);
		free_expr(c_t_lexpr);
		free_expr(c_t_uexpr);
		free_expr(h_t_lexpr);
		free_expr(h_t_uexpr);
	}
	free_neuron(neuron);
	//update_state_using_previous_layers_parallel(man,fp,numlayers);
	return;
}

void handle_lstm_layer(elina_manager_t *man, elina_abstract0_t *abs, double **weights,  double *bias, size_t d, size_t h, size_t * predecessors, bool use_area_heuristic){
    fppoly_t *fp = fppoly_of_abstract0(abs);
    fppoly_internal_t *pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
    size_t lstm_index = fp->lstm_index;
    layer_t *layer = fp->layers[lstm_index];
    neuron_t **out_neurons = fp->layers[lstm_index]->neurons;
    fp->layers[lstm_index]->predecessors = predecessors;
    size_t i;
    neuron_t * neuron = neuron_alloc();
    bool first_time_step = (layer->h_t_inf==NULL && layer->h_t_sup==NULL);
    size_t k = h + d;
    if(first_time_step){
        layer->h_t_inf = (double*)malloc(h*sizeof(double));
        layer->h_t_sup = (double*)malloc(h*sizeof(double));
        layer->c_t_inf = (double*)malloc(h*sizeof(double));
        layer->c_t_sup = (double*)malloc(h*sizeof(double));
    }

    // TODO: Fix, debug: 	for(i=0; i< h; i++){
    for(i=0; i< 1; i++){
        //printf("i = %d\n",(int)i);
        expr_t *f_t_lexpr, *i_t_lexpr, *o_t_lexpr, *c_t_lexpr;
        if(first_time_step){

            i_t_lexpr =  create_dense_expr(weights[i],bias[i],d);
            c_t_lexpr =  create_dense_expr(weights[h+i],bias[h+i],d);
            f_t_lexpr =  create_dense_expr(weights[2*h+i],bias[2*h+i],d);
            o_t_lexpr =  create_dense_expr(weights[3*h+i],bias[3*h+i],d);
        }
        else{
            expr_t * tmp1 = create_dense_expr(weights[i],bias[i],d+h);
            expr_t * tmp2 = create_dense_expr(weights[h+i],bias[h+i],d+h);
            expr_t * tmp3 = create_dense_expr(weights[2*h+i],bias[2*h+i],d+h);
            expr_t * tmp4 = create_dense_expr(weights[3*h+i],bias[3*h+i],d+h);
            i_t_lexpr = concretize_dense_sub_expr(pr, tmp1, layer->h_t_inf, layer->h_t_sup, d, d+h);
            c_t_lexpr = concretize_dense_sub_expr(pr, tmp2, layer->h_t_inf, layer->h_t_sup, d, d+h);
            f_t_lexpr = concretize_dense_sub_expr(pr, tmp3, layer->h_t_inf, layer->h_t_sup, d, d+h);
            o_t_lexpr = concretize_dense_sub_expr(pr, tmp4, layer->h_t_inf, layer->h_t_sup, d, d+h);
            free_expr(tmp1);
            free_expr(tmp2);
            free_expr(tmp3);
            free_expr(tmp4);
        }

        //expr_print(f_t_lexpr);

        //printf("computing forget...\n");
        expr_t *f_t_uexpr = copy_expr(f_t_lexpr);
        expr_t *tmp_f_t_lexpr = copy_expr(f_t_lexpr);
        expr_t *tmp_f_t_uexpr = copy_expr(f_t_uexpr);
        double lb_f_t = get_lb_using_previous_layers(man, fp, tmp_f_t_lexpr, lstm_index,use_area_heuristic);
        double ub_f_t = get_ub_using_previous_layers(man, fp, tmp_f_t_uexpr, lstm_index,use_area_heuristic);
        /* free_expr(tmp_f_t_lexpr); */
        /* free_expr(tmp_f_t_uexpr); */

        neuron->lb = lb_f_t;
        neuron->ub = ub_f_t;
        //printf("forget gate before sigmoid: lb = %lf, ub = %lf\n",neuron->lb, neuron->ub);
        //expr_print(f_t_lexpr);
        //expr_print(f_t_uexpr);
        lb_f_t = apply_sigmoid_lexpr(pr, &f_t_lexpr, neuron);
        ub_f_t = apply_sigmoid_uexpr(pr, &f_t_uexpr, neuron);
        //printf("forget gate after sigmoid: lb_f_t = %lf, ub_f_t = %lf\n",lb_f_t,ub_f_t);
        //expr_print(f_t_lexpr);
        //expr_print(f_t_uexpr);
        //printf("forget gate done\n\n");

        //printf("computing input...\n");
        expr_t *i_t_uexpr = copy_expr(i_t_lexpr);
        expr_t *tmp_i_t_lexpr = copy_expr(i_t_lexpr);
        expr_t *tmp_i_t_uexpr = copy_expr(i_t_uexpr);
        double lb_i_t = get_lb_using_previous_layers(man, fp, tmp_i_t_lexpr,lstm_index,use_area_heuristic);
        double ub_i_t = get_ub_using_previous_layers(man, fp, tmp_i_t_uexpr, lstm_index,use_area_heuristic);
        /* free_expr(tmp_i_t_lexpr); */
        /* free_expr(tmp_i_t_uexpr); */
        neuron->lb = lb_i_t;
        neuron->ub = ub_i_t;
        //printf("input gate before sigmoid: lb = %lf, ub = %lf\n",neuron->lb, neuron->ub);
        //expr_print(i_t_uexpr);
        lb_i_t = apply_sigmoid_lexpr(pr, &i_t_lexpr, neuron);
        ub_i_t = apply_sigmoid_uexpr(pr, &i_t_uexpr, neuron);
        //expr_print(i_t_uexpr);
        //printf("input gate after sigmoid: lb_i_t = %lf, ub_i_t = %lf\n",lb_i_t,ub_i_t);
        //printf("input gate done\n\n");

        //printf("computing output...\n");
        expr_t *o_t_uexpr = copy_expr(o_t_lexpr);
        expr_t *tmp_o_t_lexpr = copy_expr(o_t_lexpr);
        expr_t *tmp_o_t_uexpr = copy_expr(o_t_uexpr);
        double lb_o_t = get_lb_using_previous_layers(man, fp, tmp_o_t_lexpr, lstm_index,use_area_heuristic);
        double ub_o_t = get_ub_using_previous_layers(man, fp, tmp_o_t_uexpr, lstm_index,use_area_heuristic);
        /* free_expr(tmp_o_t_lexpr); */
        /* free_expr(tmp_o_t_uexpr); */

        neuron->lb = lb_o_t;
        neuron->ub = ub_o_t;
        //printf("output gate before sigmoid: lb = %lf, ub = %lf\n",neuron->lb, neuron->ub);
        lb_o_t = apply_sigmoid_lexpr(pr, &o_t_lexpr, neuron);
        ub_o_t = apply_sigmoid_uexpr(pr, &o_t_uexpr, neuron);
        //printf("output gate after sigmoid: lb = %lf, ub = %lf\n",lb_o_t,ub_o_t);
        out_neurons[i]->lb = lb_o_t;
        out_neurons[i]->ub = ub_o_t;
        out_neurons[i]->lexpr = o_t_lexpr;
        out_neurons[i]->uexpr = o_t_uexpr;
        //printf("output gate done\n\n");

        //printf("computing control state...\n");
        //printf("control expression:\n");
        //expr_print(c_t_lexpr);
        //printf("...\n");
        expr_t *c_t_uexpr = copy_expr(c_t_lexpr);
        expr_t *tmp_c_t_lexpr = copy_expr(c_t_lexpr);
        expr_t *tmp_c_t_uexpr = copy_expr(c_t_uexpr);
        double lb_c_t = get_lb_using_previous_layers(man, fp, tmp_c_t_lexpr, lstm_index,use_area_heuristic);
        double ub_c_t = get_ub_using_previous_layers(man, fp, tmp_c_t_uexpr, lstm_index,use_area_heuristic);
        neuron->lb = lb_c_t;
        neuron->ub = ub_c_t;
        //expr_print(c_t_lexpr);
        //expr_print(c_t_uexpr);
        //printf("control before tanh: lb = %lf, ub = %lf\n",neuron->lb,neuron->ub);
        lb_c_t = apply_tanh_lexpr(pr,&c_t_lexpr, neuron);
        ub_c_t = apply_tanh_uexpr(pr,&c_t_uexpr, neuron);
        //printf("control after tanh: lb = %lf, ub = %lf\n",lb_c_t,ub_c_t);
        //printf("control expression:\n");
        //expr_print(c_t_lexpr);
        //expr_print(c_t_uexpr);

        //printf("=======================\n");

        //printf("multiplying control by input:\n");
        expr_t *tmp_l, *tmp_u;
        double width1 = ub_i_t + lb_i_t;
        double width2 = ub_c_t + lb_c_t;
        tmp_l = c_t_lexpr;
        tmp_u = c_t_uexpr;
        //printf("control: [%lf %lf], input: [%lf %lf]\n",lb_c_t,ub_c_t,lb_i_t,ub_i_t);
        //printf("control before multiplying by input:\n");
        //expr_print(c_t_lexpr);
        //expr_print(c_t_uexpr);
        if(width1 < width2){
            //printf("concretize input\n");
            c_t_lexpr = multiply_expr(pr,c_t_lexpr,lb_i_t,ub_i_t);
            c_t_uexpr = multiply_expr(pr,c_t_uexpr,lb_i_t,ub_i_t);
        }
        else{
            //printf("concretize control\n");
            if(lb_c_t<0){
                c_t_lexpr = multiply_expr(pr,i_t_lexpr,lb_c_t,ub_c_t);
                c_t_uexpr = multiply_expr(pr,i_t_uexpr,lb_c_t,ub_c_t);
            }
            else if(ub_c_t<0){
                c_t_lexpr = multiply_expr(pr,i_t_uexpr,lb_c_t,ub_c_t);
                c_t_uexpr = multiply_expr(pr,i_t_lexpr,lb_c_t,ub_c_t);
            }
            else{
                c_t_lexpr = multiply_expr(pr,i_t_lexpr,0,0);
                c_t_uexpr = multiply_expr(pr,i_t_uexpr,0,0);
                double tmp1, tmp2;
                elina_double_interval_mul_expr_coeff(pr,&tmp1,&tmp2,lb_i_t,ub_i_t,lb_c_t,ub_c_t);
                c_t_lexpr->inf_cst += tmp1;
                c_t_lexpr->sup_cst += tmp2;
                c_t_uexpr->inf_cst += tmp1;
                c_t_uexpr->sup_cst += tmp2;
            }
        }

        //printf("control after multiplying by input:\n");
        //expr_print(c_t_lexpr);
        //expr_print(c_t_uexpr);

        free_expr(tmp_l);
        free_expr(tmp_u);

        //printf("here\n\n\n");
        //printf("====================================\n");

        if(!first_time_step){
            if(layer->c_t_inf[i]<0){
                tmp_l = multiply_expr(pr,f_t_lexpr,layer->c_t_inf[i],layer->c_t_sup[i]);
                tmp_u = multiply_expr(pr,f_t_uexpr,layer->c_t_inf[i],layer->c_t_sup[i]);
            }
            else if(layer->c_t_sup[i]<0){
                tmp_l = multiply_expr(pr,f_t_uexpr,layer->c_t_inf[i],layer->c_t_sup[i]);
                tmp_u = multiply_expr(pr,f_t_lexpr,layer->c_t_inf[i],layer->c_t_sup[i]);
            }
            else{
                tmp_l = multiply_expr(pr,f_t_lexpr,0,0);
                tmp_u = multiply_expr(pr,f_t_uexpr,0,0);
                double tmp1, tmp2;
                elina_double_interval_mul_expr_coeff(pr,&tmp1,&tmp2,lb_f_t,ub_f_t,layer->c_t_inf[i],layer->c_t_sup[i]);
                tmp_l->inf_cst += tmp1;
                tmp_l->sup_cst += tmp2;
                tmp_u->inf_cst += tmp1;
                tmp_u->sup_cst += tmp2;
            }
            add_expr(pr,c_t_lexpr,tmp_l);
            add_expr(pr,c_t_uexpr,tmp_u);
            free_expr(tmp_l);
            free_expr(tmp_u);
        }
        layer->c_t_inf[i] = get_lb_using_previous_layers(man, fp, c_t_lexpr, lstm_index,use_area_heuristic);
        layer->c_t_sup[i] = get_ub_using_previous_layers(man, fp, c_t_uexpr, lstm_index,use_area_heuristic);

        neuron->lb = layer->c_t_inf[i];
        neuron->ub = layer->c_t_sup[i];

        //printf("c_t ---> lb = %lf, ub = %lf\n", neuron->lb, neuron->ub);

        lb_c_t = apply_tanh_lexpr(pr,&c_t_lexpr, neuron);
        ub_c_t = apply_tanh_uexpr(pr,&c_t_uexpr, neuron);

        width1 = ub_o_t + lb_o_t;
        width2 = ub_c_t + lb_c_t;

        expr_t * h_t_lexpr, *h_t_uexpr;
        if(width1<width2){
            h_t_lexpr = multiply_expr(pr,c_t_lexpr,lb_o_t,ub_o_t);
            h_t_uexpr = multiply_expr(pr,c_t_uexpr,lb_o_t,ub_o_t);
        }
        else{
            h_t_lexpr =  multiply_expr(pr,o_t_lexpr,lb_c_t,ub_c_t);
            h_t_uexpr =  multiply_expr(pr,o_t_uexpr,lb_c_t,ub_c_t);
        }

        layer->h_t_inf[i] = get_lb_using_previous_layers(man, fp, h_t_lexpr, lstm_index,use_area_heuristic);
        layer->h_t_sup[i] = get_ub_using_previous_layers(man, fp, h_t_uexpr, lstm_index,use_area_heuristic);

        free_expr(f_t_lexpr);
        free_expr(f_t_uexpr);
        free_expr(i_t_lexpr);
        free_expr(i_t_uexpr);
        free_expr(c_t_lexpr);
        free_expr(c_t_uexpr);
        free_expr(h_t_lexpr);
        free_expr(h_t_uexpr);
    }
    free_neuron(neuron);
    //update_state_using_previous_layers_parallel(man,fp,numlayers);
    return;
}

#if HAS_LSTM
void lstm_handle_first_layer_(elina_manager_t *man, elina_abstract0_t *abs, double **weights,  double *bias, size_t * dim, size_t d, size_t h, size_t * predecessors, bool use_area_heuristic){
    fppoly_t *fp = fppoly_of_abstract0(abs);

    fppoly_internal_t *pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);

    create_first_lstm_layer(man, abs, h, predecessors);

    size_t lstm_index = fp->lstm_index;
    layer_t *layer = fp->layers[lstm_index];
    layer_t *previous_layer = 0;
    neuron_t **out_neurons = fp->layers[lstm_index]->neurons;
    fp->layers[lstm_index]->predecessors = predecessors;
    size_t i;
    neuron_t * neuron = neuron_alloc();
    //bool first_time_step = (layer->h_t_inf==NULL && layer->h_t_sup==NULL);
    bool first_time_step = 0;
    size_t k = h + d;
    if (lstm_index == 1)
        first_time_step = 1;
    else
        previous_layer = fp->layers[lstm_index - 1];
    //if(first_time_step)
    {

        layer->h_t_inf = (double*)malloc(h*sizeof(double));
        layer->h_t_sup = (double*)malloc(h*sizeof(double));
        layer->c_t_inf = (double*)malloc(h*sizeof(double));
        layer->c_t_sup = (double*)malloc(h*sizeof(double));
    }

    // TODO: Fix, debug: 	for(i=0; i< h; i++){
    for(i = 0; i < h; i++){
        //printf("i = %d\n",(int)i);
        expr_t *f_t_lexpr, *i_t_lexpr, *o_t_lexpr, *c_t_lexpr;
        if(first_time_step){
#if HAS_LSTM
            i_t_lexpr =  create_sparse_expr(weights[i],bias[i], dim, d);
            c_t_lexpr =  create_sparse_expr(weights[h+i],bias[h+i],dim, d);
            f_t_lexpr =  create_sparse_expr(weights[2*h+i],bias[2*h+i], dim, d);
            o_t_lexpr =  create_sparse_expr(weights[3*h+i],bias[3*h+i], dim, d);
#else
            i_t_lexpr =  create_dense_expr(weights[i],bias[i],d);
			c_t_lexpr =  create_dense_expr(weights[h+i],bias[h+i],d);
			f_t_lexpr =  create_dense_expr(weights[2*h+i],bias[2*h+i],d);
			o_t_lexpr =  create_dense_expr(weights[3*h+i],bias[3*h+i],d);
#endif
        }
        else{
#if HAS_LSTM
            i_t_lexpr =  create_sparse_expr(weights[i],bias[i], dim, d);
            c_t_lexpr =  create_sparse_expr(weights[h+i],bias[h+i],dim, d);
            f_t_lexpr =  create_sparse_expr(weights[2*h+i],bias[2*h+i], dim, d);
            o_t_lexpr =  create_sparse_expr(weights[3*h+i],bias[3*h+i], dim, d);
#if 0
            expr_t * tmp1 = create_sparse_expr(weights[i],bias[i], dim, d+h);
            expr_t * tmp2 = create_sparse_expr(weights[h+i],bias[h+i],dim, d+h);
            expr_t * tmp3 = create_sparse_expr(weights[2*h+i],bias[2*h+i], dim, d+h);
            expr_t * tmp4 = create_sparse_expr(weights[3*h+i],bias[3*h+i], dim, d+h);
            //todo
            i_t_lexpr = concretize_sparse_sub_expr(pr, tmp1, layer->h_t_inf, layer->h_t_sup, dim, d, d+h);
            c_t_lexpr = concretize_sparse_sub_expr(pr, tmp2, layer->h_t_inf, layer->h_t_sup, dim, d, d+h);
            f_t_lexpr = concretize_sparse_sub_expr(pr, tmp3, layer->h_t_inf, layer->h_t_sup, dim, d, d+h);
            o_t_lexpr = concretize_sparse_sub_expr(pr, tmp4, layer->h_t_inf, layer->h_t_sup, dim, d, d+h);
#endif
#else
            expr_t * tmp1 = create_dense_expr(weights[i],bias[i],d+h);
			expr_t * tmp2 = create_dense_expr(weights[h+i],bias[h+i],d+h);
			expr_t * tmp3 = create_dense_expr(weights[2*h+i],bias[2*h+i],d+h);
			expr_t * tmp4 = create_dense_expr(weights[3*h+i],bias[3*h+i],d+h);
			i_t_lexpr = concretize_dense_sub_expr(pr, tmp1, layer->h_t_inf, layer->h_t_sup, d, d+h);
			c_t_lexpr = concretize_dense_sub_expr(pr, tmp2, layer->h_t_inf, layer->h_t_sup, d, d+h);
			f_t_lexpr = concretize_dense_sub_expr(pr, tmp3, layer->h_t_inf, layer->h_t_sup, d, d+h);
			o_t_lexpr = concretize_dense_sub_expr(pr, tmp4, layer->h_t_inf, layer->h_t_sup, d, d+h);
#endif
#if 0
            free_expr(tmp1);
            free_expr(tmp2);
            free_expr(tmp3);
            free_expr(tmp4);
#endif
        }

        //expr_print(f_t_lexpr);

        //printf("computing forget...\n");
        expr_t *f_t_uexpr = copy_expr(f_t_lexpr);
        expr_t *tmp_f_t_lexpr = copy_expr(f_t_lexpr);
        expr_t *tmp_f_t_uexpr = copy_expr(f_t_uexpr);
        double lb_f_t = get_lb_using_previous_layers(man, fp, tmp_f_t_lexpr, lstm_index,use_area_heuristic);
        double ub_f_t = get_ub_using_previous_layers(man, fp, tmp_f_t_uexpr, lstm_index,use_area_heuristic);
        /* free_expr(tmp_f_t_lexpr); */
        /* free_expr(tmp_f_t_uexpr); */

        neuron->lb = lb_f_t;
        neuron->ub = ub_f_t;
        //printf("forget gate before sigmoid: lb = %lf, ub = %lf\n",neuron->lb, neuron->ub);
        //expr_print(f_t_lexpr);
        //expr_print(f_t_uexpr);
        lb_f_t = apply_sigmoid_lexpr(pr, &f_t_lexpr, neuron);
        ub_f_t = apply_sigmoid_uexpr(pr, &f_t_uexpr, neuron);
        //printf("forget gate after sigmoid: lb_f_t = %lf, ub_f_t = %lf\n",lb_f_t,ub_f_t);
        //expr_print(f_t_lexpr);
        //expr_print(f_t_uexpr);
        //printf("forget gate done\n\n");

        //printf("computing input...\n");
        expr_t *i_t_uexpr = copy_expr(i_t_lexpr);
        expr_t *tmp_i_t_lexpr = copy_expr(i_t_lexpr);
        expr_t *tmp_i_t_uexpr = copy_expr(i_t_uexpr);
        double lb_i_t = get_lb_using_previous_layers(man, fp, tmp_i_t_lexpr,lstm_index,use_area_heuristic);
        double ub_i_t = get_ub_using_previous_layers(man, fp, tmp_i_t_uexpr, lstm_index,use_area_heuristic);
        /* free_expr(tmp_i_t_lexpr); */
        /* free_expr(tmp_i_t_uexpr); */
        neuron->lb = lb_i_t;
        neuron->ub = ub_i_t;
        //printf("input gate before sigmoid: lb = %lf, ub = %lf\n",neuron->lb, neuron->ub);
        //expr_print(i_t_uexpr);
        lb_i_t = apply_sigmoid_lexpr(pr, &i_t_lexpr, neuron);
        ub_i_t = apply_sigmoid_uexpr(pr, &i_t_uexpr, neuron);
        //expr_print(i_t_uexpr);
        //printf("input gate after sigmoid: lb_i_t = %lf, ub_i_t = %lf\n",lb_i_t,ub_i_t);
        //printf("input gate done\n\n");

        //printf("computing output...\n");
        expr_t *o_t_uexpr = copy_expr(o_t_lexpr);
        expr_t *tmp_o_t_lexpr = copy_expr(o_t_lexpr);
        expr_t *tmp_o_t_uexpr = copy_expr(o_t_uexpr);
        double lb_o_t = get_lb_using_previous_layers(man, fp, tmp_o_t_lexpr, lstm_index,use_area_heuristic);
        double ub_o_t = get_ub_using_previous_layers(man, fp, tmp_o_t_uexpr, lstm_index,use_area_heuristic);
        /* free_expr(tmp_o_t_lexpr); */
        /* free_expr(tmp_o_t_uexpr); */

        neuron->lb = lb_o_t;
        neuron->ub = ub_o_t;
        //printf("output gate before sigmoid: lb = %lf, ub = %lf\n",neuron->lb, neuron->ub);
        lb_o_t = apply_sigmoid_lexpr(pr, &o_t_lexpr, neuron);
        ub_o_t = apply_sigmoid_uexpr(pr, &o_t_uexpr, neuron);
        //printf("output gate after sigmoid: lb = %lf, ub = %lf\n",lb_o_t,ub_o_t);
        //out_neurons[i]->lb = lb_o_t;
        //out_neurons[i]->ub = ub_o_t;
        //out_neurons[i]->lexpr = o_t_lexpr;
        //out_neurons[i]->uexpr = o_t_uexpr;
        //printf("output gate done\n\n");

        //printf("computing control state...\n");
        //printf("control expression:\n");
        //expr_print(c_t_lexpr);
        //printf("...\n");
        expr_t *c_t_uexpr = copy_expr(c_t_lexpr);
        expr_t *tmp_c_t_lexpr = copy_expr(c_t_lexpr);
        expr_t *tmp_c_t_uexpr = copy_expr(c_t_uexpr);
        double lb_c_t = get_lb_using_previous_layers(man, fp, tmp_c_t_lexpr, lstm_index,use_area_heuristic);
        double ub_c_t = get_ub_using_previous_layers(man, fp, tmp_c_t_uexpr, lstm_index,use_area_heuristic);
        neuron->lb = lb_c_t;
        neuron->ub = ub_c_t;
        //expr_print(c_t_lexpr);
        //expr_print(c_t_uexpr);
        //printf("control before tanh: lb = %lf, ub = %lf\n",neuron->lb,neuron->ub);
        lb_c_t = apply_tanh_lexpr(pr,&c_t_lexpr, neuron);
        ub_c_t = apply_tanh_uexpr(pr,&c_t_uexpr, neuron);
        //printf("control after tanh: lb = %lf, ub = %lf\n",lb_c_t,ub_c_t);
        //printf("control expression:\n");
        //expr_print(c_t_lexpr);
        //expr_print(c_t_uexpr);

        //printf("=======================\n");

        //printf("multiplying control by input:\n");
        expr_t *tmp_l, *tmp_u;
        double width1 = ub_i_t + lb_i_t;
        double width2 = ub_c_t + lb_c_t;
        tmp_l = copy_expr(c_t_lexpr);
        tmp_u = copy_expr(c_t_uexpr);
        //printf("control: [%lf %lf], input: [%lf %lf]\n",lb_c_t,ub_c_t,lb_i_t,ub_i_t);
        //printf("control before multiplying by input:\n");
        //expr_print(c_t_lexpr);
        //expr_print(c_t_uexpr);
        if(width1 < width2){
            //printf("concretize input\n");
            c_t_lexpr = multiply_expr(pr,tmp_l,lb_i_t,ub_i_t);
            c_t_uexpr = multiply_expr(pr,tmp_u,lb_i_t,ub_i_t);
        }
        else{
            //printf("concretize control\n");
            if(lb_c_t<0){
                c_t_lexpr = multiply_expr(pr,i_t_lexpr,lb_c_t,ub_c_t);
                c_t_uexpr = multiply_expr(pr,i_t_uexpr,lb_c_t,ub_c_t);
            }
            else if(ub_c_t<0){
                c_t_lexpr = multiply_expr(pr,i_t_uexpr,lb_c_t,ub_c_t);
                c_t_uexpr = multiply_expr(pr,i_t_lexpr,lb_c_t,ub_c_t);
            }
            else{
                c_t_lexpr = multiply_expr(pr,i_t_lexpr,0,0);
                c_t_uexpr = multiply_expr(pr,i_t_uexpr,0,0);
                double tmp1, tmp2;
                elina_double_interval_mul_expr_coeff(pr,&tmp1,&tmp2,lb_i_t,ub_i_t,lb_c_t,ub_c_t);
                c_t_lexpr->inf_cst += tmp1;
                c_t_lexpr->sup_cst += tmp2;
                c_t_uexpr->inf_cst += tmp1;
                c_t_uexpr->sup_cst += tmp2;
            }
        }

        //printf("control after multiplying by input:\n");
        //expr_print(c_t_lexpr);
        //expr_print(c_t_uexpr);

        free_expr(tmp_l);
        free_expr(tmp_u);

        //printf("here\n\n\n");
        //printf("====================================\n");

        if(!first_time_step){
            if(previous_layer->c_t_inf[i]<0){
                tmp_l = multiply_expr(pr,f_t_lexpr,previous_layer->c_t_inf[i],previous_layer->c_t_sup[i]);
                tmp_u = multiply_expr(pr,f_t_uexpr,previous_layer->c_t_inf[i],previous_layer->c_t_sup[i]);
            }
            else if(previous_layer->c_t_sup[i]<0){
                tmp_l = multiply_expr(pr,f_t_uexpr,previous_layer->c_t_inf[i],previous_layer->c_t_sup[i]);
                tmp_u = multiply_expr(pr,f_t_lexpr,previous_layer->c_t_inf[i],previous_layer->c_t_sup[i]);
            }
            else{
                tmp_l = multiply_expr(pr,f_t_lexpr,0,0);
                tmp_u = multiply_expr(pr,f_t_uexpr,0,0);
                double tmp1, tmp2;
                elina_double_interval_mul_expr_coeff(pr,&tmp1,&tmp2,lb_f_t,ub_f_t,previous_layer->c_t_inf[i],previous_layer->c_t_sup[i]);
                tmp_l->inf_cst += tmp1;
                tmp_l->sup_cst += tmp2;
                tmp_u->inf_cst += tmp1;
                tmp_u->sup_cst += tmp2;
            }
            add_expr(pr,c_t_lexpr,tmp_l);
            add_expr(pr,c_t_uexpr,tmp_u);
            free_expr(tmp_l);
            free_expr(tmp_u);
        }
        layer->c_t_inf[i] = get_lb_using_previous_layers(man, fp, c_t_lexpr, lstm_index,use_area_heuristic);
        layer->c_t_sup[i] = get_ub_using_previous_layers(man, fp, c_t_uexpr, lstm_index,use_area_heuristic);

        neuron->lb = layer->c_t_inf[i];
        neuron->ub = layer->c_t_sup[i];

        //printf("c_t ---> lb = %lf, ub = %lf\n", neuron->lb, neuron->ub);

        lb_c_t = apply_tanh_lexpr(pr,&c_t_lexpr, neuron);
        ub_c_t = apply_tanh_uexpr(pr,&c_t_uexpr, neuron);

        width1 = ub_o_t + lb_o_t;
        width2 = ub_c_t + lb_c_t;

        expr_t * h_t_lexpr, *h_t_uexpr;
        if(width1<width2){
            h_t_lexpr = multiply_expr(pr,c_t_lexpr,lb_o_t,ub_o_t);
            h_t_uexpr = multiply_expr(pr,c_t_uexpr,lb_o_t,ub_o_t);
        }
        else{
            h_t_lexpr =  multiply_expr(pr,o_t_lexpr,lb_c_t,ub_c_t);
            h_t_uexpr =  multiply_expr(pr,o_t_uexpr,lb_c_t,ub_c_t);
        }

        layer->h_t_inf[i] = get_lb_using_previous_layers(man, fp, h_t_lexpr, lstm_index,use_area_heuristic);
        layer->h_t_sup[i] = get_ub_using_previous_layers(man, fp, h_t_uexpr, lstm_index,use_area_heuristic);

        out_neurons[i]->lb = layer->h_t_inf[i];
        out_neurons[i]->ub = layer->h_t_sup[i];
        out_neurons[i]->lexpr = h_t_lexpr;
        out_neurons[i]->uexpr = h_t_uexpr;

        //free_expr(h_t_lexpr);
        //free_expr(h_t_uexpr);
        free_expr(o_t_lexpr);
        free_expr(o_t_uexpr);
        free_expr(f_t_lexpr);
        free_expr(f_t_uexpr);
        free_expr(i_t_lexpr);
        free_expr(i_t_uexpr);
        free_expr(c_t_lexpr);
        free_expr(c_t_uexpr);

    }
    free_neuron(neuron);
    //update_state_using_previous_layers_parallel(man,fp,numlayers);
    return;
}

void lstm_add_new_layer(elina_manager_t *man, elina_abstract0_t *abs, size_t h, size_t *predecessors){
    create_lstm_layer(man, abs, h, predecessors);
    return;
}
#define TEST_MUL_OP 0
void lstm_handle_intermediate_layer_(elina_manager_t *man, elina_abstract0_t *abs, double **weights,  double *bias, size_t * dim, size_t d, size_t h, size_t * predecessors, bool use_area_heuristic) {

    fppoly_t *fp = fppoly_of_abstract0(abs);

    fppoly_internal_t *pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);

    lstm_add_new_layer(man, abs, h, predecessors);

    size_t lstm_index = fp->lstm_index;
    layer_t *layer = fp->layers[lstm_index];
    layer_t *previous_layer = 0;
    neuron_t **out_neurons = fp->layers[lstm_index]->neurons;
    fp->layers[lstm_index]->predecessors = predecessors;
    size_t i;
    neuron_t * neuron = neuron_alloc();
    //bool first_time_step = (layer->h_t_inf==NULL && layer->h_t_sup==NULL);
    bool first_time_step = 0;
    size_t k = h + d;
    if (lstm_index == 1)
        first_time_step = 1;
    else
        previous_layer = fp->layers[lstm_index - 1];
    //if(first_time_step)
    {

        layer->h_t_inf = (double*)malloc(h*sizeof(double));
        layer->h_t_sup = (double*)malloc(h*sizeof(double));
        layer->c_t_inf = (double*)malloc(h*sizeof(double));
        layer->c_t_sup = (double*)malloc(h*sizeof(double));
    }

    for(i = 0; i < h; i++){
        //printf("i = %d\n",(int)i);
        expr_t *f_t_lexpr, *i_t_lexpr, *o_t_lexpr, *c_t_lexpr;
        if(first_time_step){
#if HAS_LSTM
            i_t_lexpr =  create_sparse_expr(weights[i],bias[i], dim, d);
            f_t_lexpr =  create_sparse_expr(weights[h+i],bias[h+i],dim, d);
            c_t_lexpr =  create_sparse_expr(weights[2*h+i],bias[2*h+i], dim, d);
            o_t_lexpr =  create_sparse_expr(weights[3*h+i],bias[3*h+i], dim, d);
#else
            i_t_lexpr =  create_dense_expr(weights[i],bias[i],d);
			c_t_lexpr =  create_dense_expr(weights[h+i],bias[h+i],d);
			f_t_lexpr =  create_dense_expr(weights[2*h+i],bias[2*h+i],d);
			o_t_lexpr =  create_dense_expr(weights[3*h+i],bias[3*h+i],d);
#endif
        }
        else{
#if HAS_LSTM
            i_t_lexpr =  create_sparse_expr(weights[i],bias[i], dim, d);
            f_t_lexpr =  create_sparse_expr(weights[h+i],bias[h+i],dim, d);
            c_t_lexpr =  create_sparse_expr(weights[2*h+i],bias[2*h+i], dim, d);
            o_t_lexpr =  create_sparse_expr(weights[3*h+i],bias[3*h+i], dim, d);
#if 0
            expr_t * tmp1 = create_sparse_expr(weights[i],bias[i], dim, d+h);
            expr_t * tmp2 = create_sparse_expr(weights[h+i],bias[h+i],dim, d+h);
            expr_t * tmp3 = create_sparse_expr(weights[2*h+i],bias[2*h+i], dim, d+h);
            expr_t * tmp4 = create_sparse_expr(weights[3*h+i],bias[3*h+i], dim, d+h);
            //todo
            i_t_lexpr = concretize_sparse_sub_expr(pr, tmp1, layer->h_t_inf, layer->h_t_sup, dim, d, d+h);
            c_t_lexpr = concretize_sparse_sub_expr(pr, tmp2, layer->h_t_inf, layer->h_t_sup, dim, d, d+h);
            f_t_lexpr = concretize_sparse_sub_expr(pr, tmp3, layer->h_t_inf, layer->h_t_sup, dim, d, d+h);
            o_t_lexpr = concretize_sparse_sub_expr(pr, tmp4, layer->h_t_inf, layer->h_t_sup, dim, d, d+h);
#endif
#else
            expr_t * tmp1 = create_dense_expr(weights[i],bias[i],d+h);
			expr_t * tmp2 = create_dense_expr(weights[h+i],bias[h+i],d+h);
			expr_t * tmp3 = create_dense_expr(weights[2*h+i],bias[2*h+i],d+h);
			expr_t * tmp4 = create_dense_expr(weights[3*h+i],bias[3*h+i],d+h);
			i_t_lexpr = concretize_dense_sub_expr(pr, tmp1, layer->h_t_inf, layer->h_t_sup, d, d+h);
			c_t_lexpr = concretize_dense_sub_expr(pr, tmp2, layer->h_t_inf, layer->h_t_sup, d, d+h);
			f_t_lexpr = concretize_dense_sub_expr(pr, tmp3, layer->h_t_inf, layer->h_t_sup, d, d+h);
			o_t_lexpr = concretize_dense_sub_expr(pr, tmp4, layer->h_t_inf, layer->h_t_sup, d, d+h);
#endif
#if 0
            free_expr(tmp1);
            free_expr(tmp2);
            free_expr(tmp3);
            free_expr(tmp4);
#endif
        }

        //expr_print(f_t_lexpr);
        //printf("computing forget...\n");
        expr_t *f_t_uexpr = copy_expr(f_t_lexpr);
        expr_t *tmp_f_t_lexpr = copy_expr(f_t_lexpr);
        expr_t *tmp_f_t_uexpr = copy_expr(f_t_uexpr);

        double lb_f_t = get_lb_using_previous_layers(man, fp, tmp_f_t_lexpr, lstm_index,use_area_heuristic);
        double ub_f_t = get_ub_using_previous_layers(man, fp, tmp_f_t_uexpr, lstm_index,use_area_heuristic);

        /* free_expr(tmp_f_t_lexpr); */
        /* free_expr(tmp_f_t_uexpr); */

        neuron->lb = lb_f_t;
        neuron->ub = ub_f_t;
        //printf("forget gate before sigmoid: lb = %lf, ub = %lf\n",neuron->lb, neuron->ub);
        //expr_print(f_t_lexpr);
        //expr_print(f_t_uexpr);

        lb_f_t = apply_sigmoid_lexpr(pr, &f_t_lexpr, neuron);
        ub_f_t = apply_sigmoid_uexpr(pr, &f_t_uexpr, neuron);

        //printf("forget gate after sigmoid: lb_f_t = %lf, ub_f_t = %lf\n",lb_f_t,ub_f_t);
        //expr_print(f_t_lexpr);
        //expr_print(f_t_uexpr);
        //printf("forget gate done\n\n");

        //printf("computing input...\n");
        expr_t *i_t_uexpr = copy_expr(i_t_lexpr);
        expr_t *tmp_i_t_lexpr = copy_expr(i_t_lexpr);
        expr_t *tmp_i_t_uexpr = copy_expr(i_t_uexpr);

        double lb_i_t = get_lb_using_previous_layers(man, fp, tmp_i_t_lexpr,lstm_index,use_area_heuristic);
        double ub_i_t = get_ub_using_previous_layers(man, fp, tmp_i_t_uexpr, lstm_index,use_area_heuristic);

        /* free_expr(tmp_i_t_lexpr); */
        /* free_expr(tmp_i_t_uexpr); */
        neuron->lb = lb_i_t;
        neuron->ub = ub_i_t;
        //printf("input gate before sigmoid: lb = %lf, ub = %lf\n",neuron->lb, neuron->ub);
        //expr_print(i_t_uexpr);

        lb_i_t = apply_sigmoid_lexpr(pr, &i_t_lexpr, neuron);
        ub_i_t = apply_sigmoid_uexpr(pr, &i_t_uexpr, neuron);

        //expr_print(i_t_uexpr);
        //printf("input gate after sigmoid: lb_i_t = %lf, ub_i_t = %lf\n",lb_i_t,ub_i_t);
        //printf("input gate done\n\n");

        //printf("computing output...\n");
        expr_t *o_t_uexpr = copy_expr(o_t_lexpr);
        expr_t *tmp_o_t_lexpr = copy_expr(o_t_lexpr);
        expr_t *tmp_o_t_uexpr = copy_expr(o_t_uexpr);

        double lb_o_t = get_lb_using_previous_layers(man, fp, tmp_o_t_lexpr, lstm_index,use_area_heuristic);
        double ub_o_t = get_ub_using_previous_layers(man, fp, tmp_o_t_uexpr, lstm_index,use_area_heuristic);

        /* free_expr(tmp_o_t_lexpr); */
        /* free_expr(tmp_o_t_uexpr); */

        neuron->lb = lb_o_t;
        neuron->ub = ub_o_t;
        //printf("output gate before sigmoid: lb = %lf, ub = %lf\n",neuron->lb, neuron->ub);

        lb_o_t = apply_sigmoid_lexpr(pr, &o_t_lexpr, neuron);
        ub_o_t = apply_sigmoid_uexpr(pr, &o_t_uexpr, neuron);

        //printf("output gate after sigmoid: lb = %lf, ub = %lf\n",lb_o_t,ub_o_t);
        //out_neurons[i]->lb = lb_o_t;
        //out_neurons[i]->ub = ub_o_t;
        //out_neurons[i]->lexpr = o_t_lexpr;
        //out_neurons[i]->uexpr = o_t_uexpr;
        //printf("output gate done\n\n");

        //printf("computing control state...\n");
        //printf("control expression:\n");
        //expr_print(c_t_lexpr);
        //printf("...\n");
        expr_t *c_t_uexpr = copy_expr(c_t_lexpr);
        expr_t *tmp_c_t_lexpr = copy_expr(c_t_lexpr);
        expr_t *tmp_c_t_uexpr = copy_expr(c_t_uexpr);

        double lb_c_t = get_lb_using_previous_layers(man, fp, tmp_c_t_lexpr, lstm_index,use_area_heuristic);
        double ub_c_t = get_ub_using_previous_layers(man, fp, tmp_c_t_uexpr, lstm_index,use_area_heuristic);

        neuron->lb = lb_c_t;
        neuron->ub = ub_c_t;
        //expr_print(c_t_lexpr);
        //expr_print(c_t_uexpr);
        //printf("control before tanh: lb = %lf, ub = %lf\n",neuron->lb,neuron->ub);

        lb_c_t = apply_tanh_lexpr(pr,&c_t_lexpr, neuron);
        ub_c_t = apply_tanh_uexpr(pr,&c_t_uexpr, neuron);

        //printf("control after tanh: lb = %lf, ub = %lf\n",lb_c_t,ub_c_t);
        //printf("control expression:\n");
        //expr_print(c_t_lexpr);
        //expr_print(c_t_uexpr);

        //printf("=======================\n");

        //printf("multiplying control by input:\n");
#if TEST_MUL_OP   //sunbing test multipication: ct = ct + it
        expr_t *tmp_l, *tmp_u;
        add_expr(pr,c_t_lexpr,i_t_lexpr);
        add_expr(pr,c_t_uexpr,i_t_uexpr);

        // ct = ct + ft
        add_expr(pr,c_t_lexpr,f_t_lexpr);
        add_expr(pr,c_t_uexpr,f_t_uexpr);
#else
        expr_t *tmp_l, *tmp_u;
        double width1 = (ub_i_t + lb_i_t);
        double width2 = (ub_c_t + lb_c_t);
        tmp_l = copy_expr(c_t_lexpr);
        tmp_u = copy_expr(c_t_uexpr);
        //printf("control: [%lf %lf], input: [%lf %lf]\n",lb_c_t,ub_c_t,lb_i_t,ub_i_t);
        //printf("control before multiplying by input:\n");
        //expr_print(c_t_lexpr);
        //expr_print(c_t_uexpr);
        if(width1 < width2){
            //printf("concretize input\n");
            c_t_lexpr = multiply_expr(pr,tmp_l,lb_i_t,ub_i_t);
            c_t_uexpr = multiply_expr(pr,tmp_u,lb_i_t,ub_i_t);
        }
        else{
            //printf("concretize control\n");
            if(lb_c_t<0){
                c_t_lexpr = multiply_expr(pr,i_t_lexpr,lb_c_t,ub_c_t);
                c_t_uexpr = multiply_expr(pr,i_t_uexpr,lb_c_t,ub_c_t);
            }
            else if(ub_c_t<0){
                c_t_lexpr = multiply_expr(pr,i_t_uexpr,lb_c_t,ub_c_t);
                c_t_uexpr = multiply_expr(pr,i_t_lexpr,lb_c_t,ub_c_t);
            }
            else{
                c_t_lexpr = multiply_expr(pr,i_t_lexpr,0,0);
                c_t_uexpr = multiply_expr(pr,i_t_uexpr,0,0);
                double tmp1, tmp2;
                elina_double_interval_mul_expr_coeff(pr,&tmp1,&tmp2,lb_i_t,ub_i_t,lb_c_t,ub_c_t);
                c_t_lexpr->inf_cst += tmp1;
                c_t_lexpr->sup_cst += tmp2;
                c_t_uexpr->inf_cst += tmp1;
                c_t_uexpr->sup_cst += tmp2;
            }
        }

        //printf("control after multiplying by input:\n");
        //expr_print(c_t_lexpr);
        //expr_print(c_t_uexpr);

        free_expr(tmp_l);
        free_expr(tmp_u);

        //printf("here\n\n\n");
        //printf("====================================\n");
        if(!first_time_step){
            if(previous_layer->c_t_inf[i]<0){
                tmp_l = multiply_expr(pr,f_t_lexpr,previous_layer->c_t_inf[i],previous_layer->c_t_sup[i]);
                tmp_u = multiply_expr(pr,f_t_uexpr,previous_layer->c_t_inf[i],previous_layer->c_t_sup[i]);
            }
            else if(previous_layer->c_t_sup[i]<0){
                tmp_l = multiply_expr(pr,f_t_uexpr,previous_layer->c_t_inf[i],previous_layer->c_t_sup[i]);
                tmp_u = multiply_expr(pr,f_t_lexpr,previous_layer->c_t_inf[i],previous_layer->c_t_sup[i]);
            }
            else{
                tmp_l = multiply_expr(pr,f_t_lexpr,0,0);
                tmp_u = multiply_expr(pr,f_t_uexpr,0,0);
                double tmp1, tmp2;
                elina_double_interval_mul_expr_coeff(pr,&tmp1,&tmp2,lb_f_t,ub_f_t,previous_layer->c_t_inf[i],previous_layer->c_t_sup[i]);
                tmp_l->inf_cst += tmp1;
                tmp_l->sup_cst += tmp2;
                tmp_u->inf_cst += tmp1;
                tmp_u->sup_cst += tmp2;
            }
            add_expr(pr,c_t_lexpr,tmp_l);
            add_expr(pr,c_t_uexpr,tmp_u);
            free_expr(tmp_l);
            free_expr(tmp_u);
        }
#endif
        layer->c_t_inf[i] = get_lb_using_previous_layers(man, fp, c_t_lexpr, lstm_index,use_area_heuristic);
        layer->c_t_sup[i] = get_ub_using_previous_layers(man, fp, c_t_uexpr, lstm_index,use_area_heuristic);

        neuron->lb = layer->c_t_inf[i];
        neuron->ub = layer->c_t_sup[i];

        //printf("c_t ---> lb = %lf, ub = %lf\n", neuron->lb, neuron->ub);

        lb_c_t = apply_tanh_lexpr(pr,&c_t_lexpr, neuron);
        ub_c_t = apply_tanh_uexpr(pr,&c_t_uexpr, neuron);
#if TEST_MUL_OP   //sunbing test multipication:
        expr_t * h_t_lexpr, *h_t_uexpr;
        h_t_lexpr = copy_expr(c_t_lexpr);
        h_t_uexpr = copy_expr(c_t_uexpr);

        add_expr(pr,h_t_lexpr,o_t_lexpr);
        add_expr(pr,h_t_uexpr,o_t_uexpr);
#else
        width1 = (ub_o_t + lb_o_t);
        width2 = (ub_c_t + lb_c_t);

        expr_t * h_t_lexpr, *h_t_uexpr;
        if(width1<width2){
            h_t_lexpr = multiply_expr(pr,c_t_lexpr,lb_o_t,ub_o_t);
            h_t_uexpr = multiply_expr(pr,c_t_uexpr,lb_o_t,ub_o_t);
        }
        else{
            //sunbing
#if 0
            h_t_lexpr =  multiply_expr(pr,o_t_lexpr,lb_c_t,ub_c_t);
            h_t_uexpr =  multiply_expr(pr,o_t_uexpr,lb_c_t,ub_c_t);
#else
            if(lb_c_t<0){
                h_t_lexpr = multiply_expr(pr,o_t_lexpr,lb_c_t,ub_c_t);
                h_t_uexpr = multiply_expr(pr,o_t_uexpr,lb_c_t,ub_c_t);
            }
            else if(ub_c_t<0){
                h_t_lexpr = multiply_expr(pr,o_t_uexpr,lb_c_t,ub_c_t);
                h_t_uexpr = multiply_expr(pr,o_t_lexpr,lb_c_t,ub_c_t);
            }
            else{
                h_t_lexpr = multiply_expr(pr,o_t_lexpr,0,0);
                h_t_uexpr = multiply_expr(pr,o_t_uexpr,0,0);
                double tmp1, tmp2;
                elina_double_interval_mul_expr_coeff(pr,&tmp1,&tmp2,lb_o_t,ub_o_t,lb_c_t,ub_c_t);
                h_t_lexpr->inf_cst += tmp1;
                h_t_lexpr->sup_cst += tmp2;
                h_t_uexpr->inf_cst += tmp1;
                h_t_uexpr->sup_cst += tmp2;
            }
#endif
        }
#endif
        layer->h_t_inf[i] = get_lb_using_previous_layers(man, fp, h_t_lexpr, lstm_index,use_area_heuristic);
        layer->h_t_sup[i] = get_ub_using_previous_layers(man, fp, h_t_uexpr, lstm_index,use_area_heuristic);

        out_neurons[i]->lb = layer->h_t_inf[i];
        out_neurons[i]->ub = layer->h_t_sup[i];
        out_neurons[i]->lexpr = h_t_lexpr;
        out_neurons[i]->uexpr = h_t_uexpr;


        //free_expr(h_t_lexpr);
        //free_expr(h_t_uexpr);
        free_expr(o_t_lexpr);
        free_expr(o_t_uexpr);
        free_expr(f_t_lexpr);
        free_expr(f_t_uexpr);
        free_expr(i_t_lexpr);
        free_expr(i_t_uexpr);
        free_expr(c_t_lexpr);
        free_expr(c_t_uexpr);

    }
    free_neuron(neuron);

    //update_state_using_previous_layers_parallel(man,fp,numlayers);
    return;

}


void lstm_handle_last_layer_(elina_manager_t *man, elina_abstract0_t *abs, double **weights,  double *bias, size_t * dim, size_t d, size_t h, size_t * predecessors, bool use_area_heuristic) {
    fppoly_t *fp = fppoly_of_abstract0(abs);
    size_t numlayers = fp->numlayers;
    fppoly_internal_t *pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);

    fppoly_add_new_layer(fp,h, FFN, NONE);

    output_abstract_t * out = (output_abstract_t*)malloc(sizeof(output_abstract_t));
    out->output_inf = (double *)malloc(h*sizeof(double));
    out->output_sup = (double *)malloc(h*sizeof(double));
    out->lexpr = (expr_t **)malloc(h*sizeof(expr_t *));
    out->uexpr = (expr_t **)malloc(h*sizeof(expr_t *));
    fp->out = out;
    neuron_t ** out_neurons = fp->layers[numlayers]->neurons;
    fp->layers[numlayers]->predecessors = predecessors;

    size_t i;

    for(i=0; i < h; i++){
        double * weight_i = weights[i];
        double bias_i = bias[i];
        //out_neurons[i]->expr = create_dense_expr(weight_i, bias_i, num_in_neurons);
        out_neurons[i]->expr = create_sparse_expr(weight_i, bias_i, dim, d);
    }

    update_state_using_previous_layers_parallel(man, fp, numlayers, use_area_heuristic);

    for(i=0; i < h; i++){
        out->output_inf[i] = out_neurons[i]->lb;
        out->output_sup[i] = out_neurons[i]->ub;
    }

    return;

}
#if 0
void gru_handle_intermediate_layer_(elina_manager_t *man, elina_abstract0_t *abs, double **weights, double *bias, size_t * dim, size_t d, size_t h, size_t * predecessors, bool use_area_heuristic) {
    fppoly_t *fp = fppoly_of_abstract0(abs);

    fppoly_internal_t *pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);

    lstm_add_new_layer(man, abs, h, predecessors);

    size_t lstm_index = fp->lstm_index;
    layer_t *layer = fp->layers[lstm_index];
    layer_t *previous_layer = 0;
    neuron_t **out_neurons = fp->layers[lstm_index]->neurons;
    fp->layers[lstm_index]->predecessors = predecessors;
    size_t i;
    neuron_t * neuron = neuron_alloc();

    bool first_time_step = 0;
    size_t k = h + d;
    if (lstm_index == 1)
        first_time_step = 1;
    else
        previous_layer = fp->layers[lstm_index - 1];

    layer->h_t_inf = (double*)malloc(h*sizeof(double));
    layer->h_t_sup = (double*)malloc(h*sizeof(double));

    for(i = 0; i < h; i++){
        //printf("i = %d\n",(int)i);
        expr_t *r_t_lexpr, *z_t_lexpr, *n_t_lexpr;
        if(first_time_step){
            r_t_lexpr =  create_sparse_expr(weights[i],bias[i], dim, d);
            z_t_lexpr =  create_sparse_expr(weights[h+i],bias[h+i],dim, d);
            n_t_lexpr =  create_sparse_expr(weights[2*h+i],bias[2*h+i], dim, d);
        }
        else{
            r_t_lexpr =  create_sparse_expr(weights[i],bias[i], dim, d);
            z_t_lexpr =  create_sparse_expr(weights[h+i],bias[h+i],dim, d);
            n_t_lexpr =  create_sparse_expr(weights[2*h+i],bias[2*h+i], dim, d);
        }

        expr_t *r_t_uexpr = copy_expr(r_t_lexpr);
        expr_t *tmp_r_t_lexpr = copy_expr(r_t_lexpr);
        expr_t *tmp_r_t_uexpr = copy_expr(r_t_lexpr);
        double lb_r_t = get_lb_using_previous_layers(man, fp, tmp_r_t_lexpr, lstm_index, use_area_heuristic);
        double ub_r_t = get_ub_using_previous_layers(man, fp, tmp_r_t_uexpr, lstm_index, use_area_heuristic);

        neuron->lb = lb_r_t;
        neuron->ub = ub_r_t;

        lb_r_t = apply_sigmoid_lexpr(pr, &r_t_lexpr, neuron);
        ub_r_t = apply_sigmoid_uexpr(pr, &r_t_uexpr, neuron);

        expr_t *z_t_uexpr = copy_expr(z_t_lexpr);
        expr_t *tmp_z_t_lexpr = copy_expr(z_t_lexpr);
        expr_t *tmp_z_t_uexpr = copy_expr(z_t_uexpr);
        double lb_z_t = get_lb_using_previous_layers(man, fp, tmp_z_t_lexpr,lstm_index,use_area_heuristic);
        double ub_z_t = get_ub_using_previous_layers(man, fp, tmp_z_t_uexpr, lstm_index,use_area_heuristic);

        neuron->lb = lb_z_t;
        neuron->ub = ub_z_t;

        lb_z_t = apply_sigmoid_lexpr(pr, &z_t_lexpr, neuron);
        ub_z_t = apply_sigmoid_uexpr(pr, &z_t_uexpr, neuron);


        // calculate r * h(t-1) first
        expr_t *tmp_l, *tmp_u;
        //if(!first_time_step)
        {
            if(previous_layer->h_t_inf[i] < 0){
                tmp_l = multiply_expr(pr,r_t_lexpr,previous_layer->h_t_inf[i],previous_layer->h_t_sup[i]);
                tmp_u = multiply_expr(pr,r_t_uexpr,previous_layer->h_t_inf[i],previous_layer->h_t_sup[i]);
            }
            else if(previous_layer->h_t_sup[i]<0){
                tmp_l = multiply_expr(pr,r_t_uexpr,previous_layer->h_t_inf[i],previous_layer->h_t_sup[i]);
                tmp_u = multiply_expr(pr,r_t_lexpr,previous_layer->h_t_inf[i],previous_layer->h_t_sup[i]);
            }
            else{
                tmp_l = multiply_expr(pr,r_t_lexpr,0,0);
                tmp_u = multiply_expr(pr,r_t_uexpr,0,0);
                double tmp1, tmp2;
                elina_double_interval_mul_expr_coeff(pr,&tmp1,&tmp2,lb_r_t,ub_r_t,previous_layer->h_t_inf[i],previous_layer->h_t_sup[i]);
                tmp_l->inf_cst += tmp1;
                tmp_l->sup_cst += tmp2;
                tmp_u->inf_cst += tmp1;
                tmp_u->sup_cst += tmp2;
            }
            free_expr(tmp_l);
            free_expr(tmp_u);
        }




        // calculate ht
        expr_t *n_t_uexpr = copy_expr(n_t_lexpr);
        expr_t *tmp_n_t_lexpr = copy_expr(n_t_lexpr);
        expr_t *tmp_n_t_uexpr = copy_expr(n_t_uexpr);
        double lb_n_t = get_lb_using_previous_layers(man, fp, tmp_n_t_lexpr, lstm_index,use_area_heuristic);
        double ub_n_t = get_ub_using_previous_layers(man, fp, tmp_n_t_uexpr, lstm_index,use_area_heuristic);

        neuron->lb = lb_n_t;
        neuron->ub = ub_n_t;

        lb_n_t = apply_tanh_lexpr(pr, &n_t_lexpr, neuron);
        ub_n_t = apply_tanh_lexpr(pr, &n_t_uexpr, neuron);


        layer->c_t_inf[i] = get_lb_using_previous_layers(man, fp, c_t_lexpr, lstm_index,use_area_heuristic);
        layer->c_t_sup[i] = get_ub_using_previous_layers(man, fp, c_t_uexpr, lstm_index,use_area_heuristic);

        neuron->lb = layer->c_t_inf[i];
        neuron->ub = layer->c_t_sup[i];

        lb_c_t = apply_tanh_lexpr(pr,&c_t_lexpr, neuron);
        ub_c_t = apply_tanh_uexpr(pr,&c_t_uexpr, neuron);

        width1 = ub_o_t + lb_o_t;
        width2 = ub_c_t + lb_c_t;

        expr_t * h_t_lexpr, *h_t_uexpr;
        if(width1<width2){
            h_t_lexpr = multiply_expr(pr,c_t_lexpr,lb_o_t,ub_o_t);
            h_t_uexpr = multiply_expr(pr,c_t_uexpr,lb_o_t,ub_o_t);
        }
        else{
            h_t_lexpr =  multiply_expr(pr,o_t_lexpr,lb_c_t,ub_c_t);
            h_t_uexpr =  multiply_expr(pr,o_t_uexpr,lb_c_t,ub_c_t);
        }

        layer->h_t_inf[i] = get_lb_using_previous_layers(man, fp, h_t_lexpr, lstm_index,use_area_heuristic);
        layer->h_t_sup[i] = get_ub_using_previous_layers(man, fp, h_t_uexpr, lstm_index,use_area_heuristic);

        out_neurons[i]->lb = layer->h_t_inf[i];
        out_neurons[i]->ub = layer->h_t_sup[i];
        out_neurons[i]->lexpr = h_t_lexpr;
        out_neurons[i]->uexpr = h_t_uexpr;

        free_expr(o_t_lexpr);
        free_expr(o_t_uexpr);
        free_expr(f_t_lexpr);
        free_expr(f_t_uexpr);
        free_expr(i_t_lexpr);
        free_expr(i_t_uexpr);
        free_expr(c_t_lexpr);
        free_expr(c_t_uexpr);
    }
    free_neuron(neuron);
    return;
}
#endif
#endif