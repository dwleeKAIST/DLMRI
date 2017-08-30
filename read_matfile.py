import scipy.io as sio

def read_matfile(filename, var_nam):
	mat=sio.loamat(filename)	
	
        return mat[var_name]
