#include "compute_bounds.h"

expr_t * replace_input_poly_cons_in_lexpr(fppoly_internal_t *pr, expr_t * expr, fppoly_t * fp){
	size_t dims = expr->size;
	size_t i,k;
	double tmp1, tmp2;
	expr_t * res;
	if(expr->type==DENSE){
		k = 0;
	}
	else{
		k = expr->dim[0];		
	}
	expr_t * mul_expr = NULL;
			
	if(expr->sup_coeff[0] <0){
		mul_expr = fp->input_uexpr[k];
	}
	else if(expr->inf_coeff[0] < 0){
		mul_expr = fp->input_lexpr[k];
	}
		
	if(mul_expr!=NULL){
		if(mul_expr->size==0){
			res = multiply_cst_expr(pr,mul_expr,expr->inf_coeff[0],expr->sup_coeff[0]);
		}
		else{
			res = multiply_expr(pr,mul_expr,expr->inf_coeff[0],expr->sup_coeff[0]);
		}
	}
		
	else{
		elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[0],expr->sup_coeff[0],fp->input_inf[k],fp->input_sup[k]);
		res = create_cst_expr(tmp1, -tmp1);
	}
	for(i=1; i < dims; i++){
		if(expr->type==DENSE){
			k = i;
		}
		else{
			k = expr->dim[i];
		}
			
		expr_t * mul_expr = NULL;
		expr_t * sum_expr = NULL;
		if(expr->sup_coeff[i] <0){
			mul_expr = fp->input_uexpr[k];
		}
		else if(expr->inf_coeff[i] <0){
			mul_expr = fp->input_lexpr[k];
		}
			
		if(mul_expr!=NULL){
			if(mul_expr->size==0){
				sum_expr = multiply_cst_expr(pr,mul_expr, expr->inf_coeff[i],expr->sup_coeff[i]);
				add_cst_expr(pr,res,sum_expr);
			}	
			else if(expr->inf_coeff[i]!=0 && expr->sup_coeff[i]!=0){
				sum_expr = multiply_expr(pr,mul_expr, expr->inf_coeff[i],expr->sup_coeff[i]);
				add_expr(pr,res,sum_expr);
			}
				//free_expr(mul_expr);
			if(sum_expr!=NULL){
				free_expr(sum_expr);
			}
		}
		else{
			elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->input_inf[k],fp->input_sup[k]);
			res->inf_cst = res->inf_cst + tmp1;
			res->sup_cst = res->sup_cst - tmp1;
		}
	}
		
	res->inf_cst = res->inf_cst + expr->inf_cst; 
	res->sup_cst = res->sup_cst + expr->sup_cst; 
	return res;
}


expr_t * replace_input_poly_cons_in_uexpr(fppoly_internal_t *pr, expr_t * expr, fppoly_t * fp){
	size_t dims = expr->size;
	size_t i,k;
	double tmp1, tmp2;
	expr_t * res;
	if(expr->type==DENSE){
		k = 0;
	}
	else{
		k = expr->dim[0];		
	}
	expr_t * mul_expr = NULL;
			
	if(expr->sup_coeff[0] <0){
		mul_expr = fp->input_lexpr[k];
	}
	else if(expr->inf_coeff[0] < 0){
		mul_expr = fp->input_uexpr[k];
	}
		
	if(mul_expr!=NULL){
		if(mul_expr->size==0){
			res = multiply_cst_expr(pr,mul_expr,expr->inf_coeff[0],expr->sup_coeff[0]);
		}
		else{
			res = multiply_expr(pr,mul_expr,expr->inf_coeff[0],expr->sup_coeff[0]);
		}
	}
	else{
		elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[0],expr->sup_coeff[0],fp->input_inf[k],fp->input_sup[k]);
		res = create_cst_expr(-tmp2, tmp2);
	}
                //printf("finish\n");
		//fflush(stdout);
	for(i=1; i < dims; i++){
		if(expr->type==DENSE){
			k = i;
		}
		else{
			k = expr->dim[i];
		}
		expr_t * mul_expr = NULL;
		expr_t * sum_expr = NULL;
		if(expr->sup_coeff[i] <0){
			mul_expr = fp->input_lexpr[k];
		}
		else if(expr->inf_coeff[i] <0){
			mul_expr = fp->input_uexpr[k];
		}
			
		if(mul_expr!=NULL){
			if(mul_expr->size==0){
				sum_expr = multiply_cst_expr(pr,mul_expr, expr->inf_coeff[i],expr->sup_coeff[i]);
				add_cst_expr(pr,res,sum_expr);
			}	
			else if(expr->inf_coeff[i]!=0 && expr->sup_coeff[i]!=0){
				sum_expr = multiply_expr(pr,mul_expr, expr->inf_coeff[i],expr->sup_coeff[i]);
				add_expr(pr,res,sum_expr);
			}
				//free_expr(mul_expr);
			if(sum_expr!=NULL){
				free_expr(sum_expr);
			}
		}
		else{
			elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->input_inf[k],fp->input_sup[k]);
			res->inf_cst = res->inf_cst - tmp2;
			res->sup_cst = res->sup_cst + tmp2;
		}
	}
	res->inf_cst = res->inf_cst + expr->inf_cst; 
	res->sup_cst = res->sup_cst + expr->sup_cst; 
	return res;
}


double compute_lb_from_expr(fppoly_internal_t *pr, expr_t * expr, fppoly_t * fp, int layerno){
	size_t i,k;
	double tmp1, tmp2;
        //printf("start\n");
        //fflush(stdout);
	if((fp->input_lexpr!=NULL) && (fp->input_uexpr!=NULL) && layerno==-1){
		expr =  replace_input_poly_cons_in_lexpr(pr, expr, fp);
	}
        //expr_print(expr);
	//fflush(stdout);
	size_t dims = expr->size;
	double res_inf = expr->inf_cst;
	if(expr->inf_coeff==NULL || expr->sup_coeff==NULL){
		return res_inf;
	}
	for(i=0; i < dims; i++){    //dim is how many terms in expression
		//if(expr->inf_coeff[i]<0){
		if(expr->type==DENSE){
			k = i;
		}
		else{
			k = expr->dim[i];
		}
#if HAS_RNN
        if (expr->type == SPARSE && (layerno != -1)) {
            if ((k + 1) > fp->layers[layerno]->dims) {
                tmp1 = expr->inf_coeff[i];
                res_inf = res_inf + tmp1;
                continue;
            }
        }
#endif
			if(layerno==-1){
				elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->input_inf[k],fp->input_sup[k]);
			}
			else{
				
				elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->layers[layerno]->neurons[k]->lb,fp->layers[layerno]->neurons[k]->ub);
			}
			//printf("tmp1: %g\n",tmp1);
			res_inf = res_inf + tmp1;
			
	}
//	printf("inf: %g\n",-res_inf);
//	fflush(stdout);
        if(fp->input_lexpr!=NULL && fp->input_uexpr!=NULL && layerno==-1){
		free_expr(expr);
	}
        //printf("finish\n");
        //fflush(stdout);
	return res_inf;
}

double compute_ub_from_expr(fppoly_internal_t *pr, expr_t * expr, fppoly_t * fp, int layerno){
	size_t i,k;
	double tmp1, tmp2;

	if((fp->input_lexpr!=NULL) && (fp->input_uexpr!=NULL) && layerno==-1){
		expr =  replace_input_poly_cons_in_uexpr(pr, expr, fp);
	}

	size_t dims = expr->size;
	double res_sup = expr->sup_cst;
	if(expr->inf_coeff==NULL || expr->sup_coeff==NULL){
		return res_sup;
	}
	for(i=0; i < dims; i++){
		//if(expr->inf_coeff[i]<0){
		if(expr->type==DENSE){
			k = i;
		}
		else{
			k = expr->dim[i];
		}		
		if(layerno==-1){
			elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->input_inf[k],fp->input_sup[k]);
		}
		else{
			elina_double_interval_mul(&tmp1,&tmp2,expr->inf_coeff[i],expr->sup_coeff[i],fp->layers[layerno]->neurons[k]->lb,fp->layers[layerno]->neurons[k]->ub);
		}
		res_sup = res_sup + tmp2;
			
	}
	//printf("sup: %g\n",res_sup);
	//fflush(stdout);
	if(fp->input_lexpr!=NULL && fp->input_uexpr!=NULL && layerno==-1){
		free_expr(expr);
	}
	return res_sup;
}


double get_lb_using_predecessor_layer(fppoly_internal_t * pr,fppoly_t *fp, expr_t **lexpr_ptr, size_t k, bool use_area_heuristic){
	expr_t * tmp_l;
	neuron_t ** aux_neurons = fp->layers[k]->neurons;
	expr_t *lexpr = *lexpr_ptr;
	double res = INFINITY;
	if(fp->layers[k]->type==FFN || fp->layers[k]->type==CONV){
			
		    if(fp->layers[k]->activation==RELU){
		        tmp_l = lexpr;
#if HAS_RNN
                lexpr = lexpr_replace_relu_bounds_(pr, lexpr, aux_neurons, fp->layers[k]->dims, use_area_heuristic);
#else
		        lexpr = lexpr_replace_relu_bounds(pr,lexpr,aux_neurons, use_area_heuristic);
#endif
		        free_expr(tmp_l);
		    }
		    else if(fp->layers[k]->activation==SIGMOID){
		        tmp_l = lexpr;
#if HAS_RNN
				lexpr = lexpr_replace_sigmoid_bounds_(pr,lexpr,aux_neurons, fp->layers[k]->dims);
#else
		        lexpr = lexpr_replace_sigmoid_bounds(pr,lexpr,aux_neurons);
#endif
		        free_expr(tmp_l);
		    }
		    else if(fp->layers[k]->activation==TANH){
		         tmp_l = lexpr;
#if HAS_RNN
				lexpr = lexpr_replace_tanh_bounds_(pr,lexpr,aux_neurons, fp->layers[k]->dims);
#else
		        lexpr = lexpr_replace_tanh_bounds(pr,lexpr,aux_neurons);
#endif
		        free_expr(tmp_l);
		    }
		
		    else if(fp->layers[k]->activation==PARABOLA){
		         tmp_l = lexpr;
		        lexpr = lexpr_replace_parabola_bounds(pr,lexpr,aux_neurons);
		        free_expr(tmp_l);
		    }		
			else if(fp->layers[k]->activation==LOG){
				tmp_l = lexpr;
		        	lexpr = lexpr_replace_log_bounds(pr,lexpr,aux_neurons);
		        	free_expr(tmp_l);
			}	
				tmp_l = lexpr;
				res = compute_lb_from_expr(pr,lexpr,fp,k);		
				*lexpr_ptr = expr_from_previous_layer(pr,lexpr, fp->layers[k]);		
				free_expr(tmp_l);
			}
		else{
			expr_t * tmp_l = lexpr;
#if HAS_LSTM
			*lexpr_ptr = lexpr_replace_maxpool_or_lstm_bounds_(pr,lexpr,aux_neurons, fp->layers[k]->dims);
#else
			*lexpr_ptr = lexpr_replace_maxpool_or_lstm_bounds(pr,lexpr,aux_neurons);
#endif
			free_expr(tmp_l);
		}
		return res;
}


double get_ub_using_predecessor_layer(fppoly_internal_t * pr,fppoly_t *fp, expr_t **uexpr_ptr, size_t k, bool use_area_heuristic){
	expr_t * tmp_u;
	neuron_t ** aux_neurons = fp->layers[k]->neurons;
	expr_t *uexpr = *uexpr_ptr;
	double res = INFINITY;

	if(fp->layers[k]->type==FFN || fp->layers[k]->type==CONV){
			
		    if(fp->layers[k]->activation==RELU){
		        tmp_u = uexpr;
		        uexpr = uexpr_replace_relu_bounds(pr,uexpr,aux_neurons, use_area_heuristic);
		        free_expr(tmp_u);
		    }
		    else if(fp->layers[k]->activation==SIGMOID){
		        tmp_u = uexpr;
			
		        uexpr = uexpr_replace_sigmoid_bounds(pr,uexpr,aux_neurons);
		        free_expr(tmp_u);
		    }
		    else if(fp->layers[k]->activation==TANH){
		         tmp_u = uexpr;
		        uexpr = uexpr_replace_tanh_bounds(pr,uexpr,aux_neurons);
		        free_expr(tmp_u);
		    }
		
		    else if(fp->layers[k]->activation==PARABOLA){
		         tmp_u = uexpr;
		        uexpr = uexpr_replace_parabola_bounds(pr,uexpr,aux_neurons);
		        free_expr(tmp_u);
		    }		
			else if(fp->layers[k]->activation==LOG){
				tmp_u = uexpr;
		        	uexpr = uexpr_replace_log_bounds(pr,uexpr,aux_neurons);
		        	free_expr(tmp_u);
			}	
				tmp_u = uexpr;		
				res = compute_ub_from_expr(pr,uexpr,fp,k);	
				*uexpr_ptr = expr_from_previous_layer(pr,uexpr, fp->layers[k]);
				free_expr(tmp_u);
			}
		else{
			expr_t * tmp_u = uexpr;
#if HAS_LSTM
            *uexpr_ptr = uexpr_replace_maxpool_or_lstm_bounds_(pr,uexpr,aux_neurons, fp->layers[k]->dims);
#else
			*uexpr_ptr = uexpr_replace_maxpool_or_lstm_bounds(pr,uexpr,aux_neurons);
#endif
			free_expr(tmp_u);
		}
		return res;
}

double get_lb_using_previous_layers(elina_manager_t *man, fppoly_t *fp, expr_t *expr, size_t layerno, bool use_area_heuristic){
	size_t i;
	int k;
	//size_t numlayers = fp->numlayers;
	expr_t * lexpr = copy_expr(expr);
        fppoly_internal_t * pr = fppoly_init_from_manager(man,ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	if(fp->numlayers==layerno){
		
		k = layerno-1;
	}
	else if(fp->layers[layerno]->type==RESIDUAL){
		k = layerno;
	}
	else{
		k = fp->layers[layerno]->predecessors[0]-1;
	}	
	double res = INFINITY;

	while(k >=0){
	
		if(fp->layers[k]->type==RESIDUAL){
				if(fp->layers[k]->activation==RELU && k!=(int)layerno){
					neuron_t ** aux_neurons = fp->layers[k]->neurons; 
					expr_t *tmp_l = lexpr;
		       			lexpr = lexpr_replace_relu_bounds(pr,lexpr,aux_neurons, use_area_heuristic);
					free_expr(tmp_l);
				}
				expr_t * lexpr_copy = copy_expr(lexpr);
				lexpr_copy->inf_cst = 0;
				lexpr_copy->sup_cst = 0;
				size_t predecessor1 = fp->layers[k]->predecessors[0]-1;
				size_t predecessor2 = fp->layers[k]->predecessors[1]-1;
				
				char * predecessor_map = (char *)calloc(k,sizeof(char));
				// Assume no nested residual layers
				int iter = fp->layers[predecessor1]->predecessors[0]-1;
				while(iter>=0){
					predecessor_map[iter] = 1;
					iter = fp->layers[iter]->predecessors[0]-1;
				}
				iter =  fp->layers[predecessor2]->predecessors[0]-1;
				int common_predecessor = 0;
				while(iter>=0){
					if(predecessor_map[iter] == 1){
						common_predecessor = iter;
						break;
					}
					iter = fp->layers[iter]->predecessors[0]-1;
				}
				
				iter = predecessor1;
				while(iter!=common_predecessor){
					get_lb_using_predecessor_layer(pr,fp, &lexpr,  iter, use_area_heuristic);
					iter = fp->layers[iter]->predecessors[0]-1;
				}
				iter =  predecessor2;
				while(iter!=common_predecessor){
					get_lb_using_predecessor_layer(pr,fp, &lexpr_copy,  iter, use_area_heuristic);
					iter = fp->layers[iter]->predecessors[0]-1;					
				}
				free(predecessor_map);
				add_expr(pr,lexpr,lexpr_copy);
				
				free_expr(lexpr_copy);
				
				// Assume at least one non-residual layer between two residual layers
				k = common_predecessor;
				
				continue;
			}
			else {
								
				 res =fmin(res,get_lb_using_predecessor_layer(pr,fp, &lexpr, k, use_area_heuristic));
				 k = fp->layers[k]->predecessors[0]-1;
				
			}
		
		
	   
			
	}

	res = fmin(res,compute_lb_from_expr(pr,lexpr,fp,-1)); 
        free_expr(lexpr);
	return res;
	
}


double get_ub_using_previous_layers(elina_manager_t *man, fppoly_t *fp, expr_t *expr, size_t layerno, bool use_area_heuristic){
	size_t i;
	int k;
	//size_t numlayers = fp->numlayers;

	expr_t * uexpr = copy_expr(expr);
	fppoly_internal_t * pr = fppoly_init_from_manager(man,ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
        
	if(fp->numlayers==layerno){
		k = layerno-1;
	}
	else if(fp->layers[layerno]->type==RESIDUAL){
		k = layerno;
	}
	else{
		k = fp->layers[layerno]->predecessors[0]-1;
	}	
	double res =INFINITY;
	while(k >=0){
		if(fp->layers[k]->type==RESIDUAL){
				if(fp->layers[k]->activation==RELU && k!=(int)layerno){
					neuron_t ** aux_neurons = fp->layers[k]->neurons; 
					expr_t *tmp_u = uexpr;
		       			uexpr = uexpr_replace_relu_bounds(pr,uexpr,aux_neurons, use_area_heuristic);
					free_expr(tmp_u);
				}
				expr_t * uexpr_copy = copy_expr(uexpr);
				uexpr_copy->inf_cst = 0;
				uexpr_copy->sup_cst = 0;
				size_t predecessor1 = fp->layers[k]->predecessors[0]-1;
				size_t predecessor2 = fp->layers[k]->predecessors[1]-1;
				
				char * predecessor_map = (char *)calloc(k,sizeof(char));
				// Assume no nested residual layers
				int iter = fp->layers[predecessor1]->predecessors[0]-1;
				while(iter>=0){
					predecessor_map[iter] = 1;
					iter = fp->layers[iter]->predecessors[0]-1;
				}
				iter =  fp->layers[predecessor2]->predecessors[0]-1;
				int common_predecessor = 0;
				while(iter>=0){
					if(predecessor_map[iter] == 1){
						common_predecessor = iter;
						break;
					}
					iter = fp->layers[iter]->predecessors[0]-1;
				}
				
				iter = predecessor1;
				while(iter!=common_predecessor){
					get_ub_using_predecessor_layer(pr,fp, &uexpr,  iter, use_area_heuristic);
					iter = fp->layers[iter]->predecessors[0]-1;
				}
				iter =  predecessor2;
				while(iter!=common_predecessor){
					get_ub_using_predecessor_layer(pr,fp, &uexpr_copy,  iter, use_area_heuristic);
					iter = fp->layers[iter]->predecessors[0]-1;					
				}
				free(predecessor_map);
				add_expr(pr,uexpr,uexpr_copy);
				
				free_expr(uexpr_copy);
				
				// Assume at least one non-residual layer between two residual layers
				k = common_predecessor;
				
				continue;
			}
			else {
				 res = fmin(res,get_ub_using_predecessor_layer(pr,fp, &uexpr, k, use_area_heuristic));
				 k = fp->layers[k]->predecessors[0]-1;
				 
			}
			
	}
		
	res = fmin(res,compute_ub_from_expr(pr,uexpr,fp,-1)); 
        free_expr(uexpr);
	return res;
	
}
