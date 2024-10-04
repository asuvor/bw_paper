import math 
import numpy as np 


from concurrent.futures import ThreadPoolExecutor
import logging
from os import listdir
from os.path import isfile, join
from random import sample
from scipy.linalg import eigh, pinvh 
from numpy import diag, sqrt


# from utls import *



# Operations on matrices

def sqrtmInv2(X):
    """Inversion of the square root of a symmetric matrix X."""
    #X = sqrtm(X)
    return(sqrtm1(pinvh(X)))


def sqrtm1(X):
    """Square root of a symmetric matrix"""
    eigval, eigvects = eigh(X)
    Y = (eigvects * sqrt(np.maximum(eigval, 0))[np.newaxis,:]).dot(eigvects.T)
    return(Y)

# Inversion of X
def MInv(X):
    """
    Matrix inversion using eigenvalue decomposition.
    """
    eigvals, eigvecs = eigh(X)
    inv_matrix = eigvecs @ diag(1 / eigvals) @ eigvecs.T
    return inv_matrix



#Generate basis

def GenONbasis(d):
    """
    Generate orthonormal basis in the space of d x d symmetric matrices.
    
    Parameters:
    - d: Dimension of the matrix space.

    Returns:
    - A list of basis elements, each represented as a dictionary with 'count' and 'u' (the matrix).
    """
    out = []
    counter = 0
    
    for i in range(d):
        ei = np.zeros(d)
        ei[i] = 1
        for j in range(i + 1):
            ej = np.zeros(d)
            ej[j] = 1
            counter += 1
            
            if i == j:
                u_matrix = np.outer(ei, ei)
            else:
                u_matrix = (np.outer(ei, ej) + np.outer(ej, ei)) / sqrt(2)
            
            out.append({'count': counter, 'u': u_matrix})
    
    return out

def GenONbasisVec(d):
    """ Генерация ортонормированного базиса """
    vectors = []
    for i in range(0, d-1):  
        v = np.zeros(d)
        v[i] = 1
        v[-1] = -1
        vectors.append(v)
    
    
    # Метод Грамма-Шмидта
    ortho_basis = []
    
    for v in vectors:
        # Процесс ортогонализации: вычитаем проекции на уже найденные вектора
        for u in ortho_basis:
            v = v - np.dot(v, u) * u
            
        # Нормализация вектора
        norm = np.linalg.norm(v)
        if norm > 1e-10:  # Проверка, что вектор не нулевой
            ortho_basis.append(v / norm)
    
    return np.array(ortho_basis)





# Other functions

def gen_weights(n):
    """
    Генерирует веса, которые не суммируются в 0.
    
    :param n: Размерность веса
    :return: Массив весов
    """
    weights = np.array([0, 0])
    while weights.sum() == 0:
        weights = 2 * np.random.binomial(n=1, p=0.5, size=n)
    return weights

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


def find_grid(true, boot, asm, steps = 1000):

    mn = min([min([min(curve) for curve in boot]), min(true), min([min(curve) for curve in asm])])
    mx = max([max([max(curve) for curve in boot]), max(true), max([max(curve) for curve in asm])])

    mn = max([mn - 0.1*mn, 0.0])
    mx = mx + 0.1*mx
    x_grid = np.linspace(mn,mx, steps) 

    return x_grid


def compute_ecdfs(data, grid):
    output = [ecdf_on_grid(curve, grid) for curve in data]
    return output

def compute_KS_stat(true, emp):
    dummy = [ks_stat_naive(true, i) for i in emp]

    return np.mean(dummy), np.var(dummy)