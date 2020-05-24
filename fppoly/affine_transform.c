#include "affine_transform.h"


expr_t * expr_from_previous_layer(fppoly_internal_t *pr, expr_t * expr, layer_t * prev_layer){
	if(expr->size==0){
		return copy_cst_expr(expr);
	}	
	if(expr->inf_coeff==NULL || expr->sup_coeff==NULL ){
		return alloc_expr();
	}
	
	neuron_t **prev_neurons = prev_layer->neurons;
	size_t out_num_neurons = prev_layer->dims;  // how many neurons in previous layer
	size_t in_num_neurons = expr->size; // how many terms in expr
	size_t i,k;
	expr_t * res;
	
    if(expr->type==DENSE){
        k = 0;
    }
    else{
        k = expr->dim[0];
    }

    if(prev_neurons[k]->expr->size==0){

        res = multiply_cst_expr(pr,prev_neurons[k]->expr,expr->inf_coeff[0],expr->sup_coeff[0]);

    }
    else{

        res = multiply_expr(pr,prev_neurons[k]->expr,expr->inf_coeff[0],expr->sup_coeff[0]);
    }
    
	for(i=1; i < in_num_neurons; i++){
		if(expr->type==DENSE){
			k = i;
		}
		else{
			k = expr->dim[i];
		}

#if HAS_RNN
        if (expr->type == SPARSE){
            if ((k + 1) > out_num_neurons) {
                res->inf_coeff[k] = res->inf_coeff[k] + expr->inf_coeff[k];
                res->sup_coeff[k] = res->sup_coeff[k] + expr->sup_coeff[k];
                continue;
            }
        }
#endif
        expr_t * mul_expr;
		if(prev_neurons[k]->expr->size==0){
			mul_expr = multiply_cst_expr(pr,prev_neurons[k]->expr, expr->inf_coeff[i],expr->sup_coeff[i]);
			add_cst_expr(pr,res,mul_expr);
			free_expr(mul_expr);
		}
		else if(expr->inf_coeff[i]!=0 || expr->sup_coeff[i]!=0){
			mul_expr = multiply_expr(pr,prev_neurons[k]->expr, expr->inf_coeff[i],expr->sup_coeff[i]);
			
			add_expr(pr,res,mul_expr);
			free_expr(mul_expr);
			
		}
		
	}
	res->inf_cst = res->inf_cst + expr->inf_cst; 
	res->sup_cst = res->sup_cst + expr->sup_cst; 
	
	return res;
}
