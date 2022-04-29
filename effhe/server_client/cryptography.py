import tenseal as ts
from torchvision import datasets
import torchvision.transforms as transforms
from effhe.constants.server_client import KERNEL_SHAPE, STRIDE

def gen_key(kind):
    '''
    Generates a key based on the type. We are using type SMALL.
    '''

    bits_scale = 26

    context = None
	
    if kind == "small":
	    context = ts.context(
	        ts.SCHEME_TYPE.CKKS,
	        poly_modulus_degree=8192,
	        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, 31]
	    )
    
    elif kind == "mid" :
    	context = ts.context(
		    ts.SCHEME_TYPE.CKKS,
		    poly_modulus_degree=8192,
		    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
		)
    else:
    	context = ts.context(
		    ts.SCHEME_TYPE.CKKS,
		    poly_modulus_degree=16384,
		    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
		)
        
    context.global_scale = pow(2, bits_scale)
    
    context.generate_galois_keys()
    
    return context

def encrypt_data(context, data):
    """
    takes the key (context) and datapoint to be encoded.
    returns encrypted data and windows_nb
    """
    data_enc, windows_nb = ts.im2col_encoding(
                                    context, 
                                    data, 
                                    KERNEL_SHAPE[0],
                                    KERNEL_SHAPE[1], 
                                    STRIDE
                                    )
    return data_enc, windows_nb

def make_public_key(private_key):
    """
    Given private key (context), returns a copy of the the public contents
    """
    public_key = private_key.copy()
    public_key.make_context_public()
    
    return public_key