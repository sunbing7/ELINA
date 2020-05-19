
from elina_auxiliary_imports import *
import ctypes
# ********************************************************************** #
# I. Datatypes
# ********************************************************************** #

#typedef enum exprtype_t{
# DENSE,
# SPARSE,
#}exprtype_t;

#typedef struct expr_t{
#	double *inf_coeff;
#	double *sup_coeff;
#	double inf_cst;
#	double sup_cst;
#	exprtype_t type;
#	size_t * dim;
#    size_t size;
#}expr_t;


class FppolyExprtypet(CtypesEnum):

    DENSE = 0
    SPARSE = 1


class FppolyExprt(Structure):
    """ Ctype representing the union field in expr_t from fppoly.h """

    _fields_ = [('inf_coeff', POINTER(c_double)),
                ('sup_coeff', POINTER(c_double)),
                ('inf_cst', c_double),
                ('sup_cst', c_double),
                ('type', c_int),
                ('dim', POINTER(c_size_t)),
                ('size', c_size_t)]


FppolyExprtPtr = POINTER(FppolyExprt)