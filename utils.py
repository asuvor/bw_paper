
# Operations on matrices

def sqrtmInv2(X):
    """Inversion of the square root of a symmetric matrix X."""
    #X = sqrtm(X)
    return(sqrtm1(scipy.linalg.pinvh(X)))


def sqrtm1(X, method='numpy.eigh'):
    """Square root of a symmetric matrix"""
    eigval, eigvects = eigh(X)
    Y = (eigvects * sqrt(np.maximum(eigval, 0))[np.newaxis,:]).dot(eigvects.T)
    return(Y)

# Inversion of X
def MInv(X):
    """
    Matrix inversion using eigenvalue decomposition.
    """
    eigvals, eigvecs = scipy.linalg.eigh(X)
    inv_matrix = eigvecs @ diag(1 / eigvals) @ eigvecs.T
    return inv_matrix





# Other functions

def subsmple(arr_list, sample_size, repl):
    """
    Subsample with replacement from a list of numpy arrays.
    
    Parameters:
    - arr_list: List of numpy arrays.
    - sample_size: Number of samples to draw with replacement.
    
    Returns:
    - A list of subsampled numpy arrays
    """
    # Randomly select indices with replacement
    indices = np.random.choice(len(arr_list), size=sample_size, replace = repl)
    
    # Return the subsampled numpy arrays
    return [arr_list[i] for i in indices]

def ecdf_on_grid(data, x_grid):
    """Compute the ECDF on a fixed x-grid"""
    n = len(data)
    ecdf_values = np.array([np.sum(data <= x) / n for x in x_grid])
    return ecdf_values


def ceil_to_decimals(number, decimal_places):
    # Calculate the scaling factor (10^decimal_places)
    factor = 10 ** decimal_places
    # Multiply the number by the scaling factor, apply ceiling, and divide back
    return math.ceil(number * factor) / factor

def ks_stat_naive(f1, f2):
    """Compute KS distance between curves"""
    return(np.max(np.abs(f1-f2))) 
