import math 
import os
import re
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


from aux_functions import *
from os import listdir
from os.path import isfile, join
from random import sample
from scipy.spatial import distance
from scipy.cluster import hierarchy as hc
from scipy.linalg import sqrtm, block_diag, pinvh, eigh
from numpy.linalg import inv, pinv
from numpy.random import normal, poisson, randint
from numpy import diagflat, eye, fill_diagonal, diag, sqrt
from sklearn.utils import resample
 





#------ Wasserstein part --------


def BW(K1, K2, method='numpy.eigh'):
    """Bures-Wasserstein distance"""
    Q = sqrtm1(K1, method)
    d = sqrt(np.maximum(0, K1.trace() + K2.trace() - 2 * sqrtm1(Q.dot(K2).dot(Q), method).trace()))
    return d


def Frob(K1, K2):
    """Frobenius distance"""
    d = np.linalg.norm(K1 - K2)
    return d

def Fbarycenter(covs, weights = None):
    """Weighted Frobenius mean"""
    n = len(covs)
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.array(weights) / np.sum(weights)  
    covs = [weights[i] * covs[i] for i in range(0, n)]
    return np.sum(covs, axis=0)

def Wbarycenter(covs, weights = None, eps = 1e-3, init = None, max_iterations = 1e3, 
                reg = 0, crit = 0, verbose = False):
    """Weighted Wasserstein barycenter"""
    if weights is None:
        n = len(covs)
        weights = np.ones(n) / n
    else:
        weights = np.array(weights) / np.sum(weights)    
    dim = covs[0].shape[0] # Dimension of the Gaussians
    if init is None:
        Q = eye(dim)
    else:
        Q = np.copy(init)
    # Barycenter is a solution of a fixed-point equation
    i = 0
    while True:
        Q_sqrt = sqrtm1(Q)
        Q_sqrtinv = sqrtmInv2(Q)

        T_mean = np.zeros([dim, dim], dtype = np.float64)
        for w, cov in zip(weights, covs):
            if (reg > 0):
                cov = cov + reg * eye(dim)
            else:
                cov = cov
            if w > 0:
                T_mean += w * sqrtm1(Q_sqrt.dot(cov).dot(Q_sqrt))
        tmp = Q_sqrtinv.dot(T_mean)
        T_mean = tmp.dot(Q_sqrtinv)
        Q_prev = Q
        Q = tmp.dot(tmp.T)
        if (crit == 0):
            norm_value = np.linalg.norm(T_mean - eye(dim), ord='fro')
        elif (crit == 1):
            norm_value = np.linalg.norm(Q - Q_prev, ord='fro')
        i += 1
        if verbose:
            print(i, norm_value)
        if norm_value < eps:
            if verbose:
                print('Success! Iterations number: ' + str(i))
            break
        if i > max_iterations:
            if verbose:
                print('Iterations number exceeded!')
            break
        if (reg > 0):
            Q = Q - reg * eye(dim)
    return Q




#---------------- Data generation and processing-------------




def genGLFast(mtx, P):
    """
    Generate GL, project it to the orthnormal basis P and invert
    """
    fill_diagonal(mtx, 0)
    mtx = diagflat(mtx.sum(axis=1)) - mtx
    mtx = P @ mtx @ P.T
    return(pinv(mtx))






def genWG(d, p11, p22, p12, lam11, lam22, lam12):
    """
    Generates a set of weighted graphs with two communities and Poisson-distributed weights.
    
    Parameters:
    - d: Number of total nodes (default is 20).
    - p11: Edge probability within community C11 (default is 0.8).
    - p22: Edge probability within community C22 (default is 0.5).
    - p12: Edge probability between communities C11 and C22 (default is 0.2).
    - lam11: Lambda for Poisson distribution in C11 (default is 12).
    - lam22: Lambda for Poisson distribution in C22 (default is 7).
    - lam12: Lambda for Poisson distribution in C12 and C21 (default is 2).
    
    Returns:
    - A weighted adjacency matrix representing the graph.
    - The sizes of the two communities.
    """
    # Randomly determine the sizes of the two communities
    size_C11 = int(np.floor(d/2)) + np.random.randint(-2, 2)  # Random number between 5 and 10
    size_C22 = d - size_C11  # The rest of the nodes are in C22
    
    # Initialize the adjacency matrix for the weighted graph
    weighted_adjacency_matrix = np.zeros((d, d))
    
    # Fill in the edges for community C11 (intra-community)
    for i in range(size_C11):
        for j in range(i + 1, size_C11):
            if np.random.rand() < p11:
                weight = np.random.poisson(lam11)
#                 weight = 2*(np.random.randint(6) + 4)
                weighted_adjacency_matrix[i, j] = weighted_adjacency_matrix[j, i] = weight
    
    # Fill in the edges for community C22 (intra-community)
    for i in range(size_C11, d):
        for j in range(i + 1, d):
            if np.random.rand() < p22:
                weight = np.random.poisson(lam22)
#                 weight = np.random.randint(9) + 6
                weighted_adjacency_matrix[i, j] = weighted_adjacency_matrix[j, i] = weight
    
    # Fill in the edges between communities C11 and C22 (inter-community)
    for i in range(size_C11):
        for j in range(size_C11, d):
            if np.random.rand() < p12:
                weight = np.random.poisson(lam12)
#                 weight = np.random.randint(4)
                weighted_adjacency_matrix[i, j] = weighted_adjacency_matrix[j, i] = weight
    
    np.fill_diagonal(weighted_adjacency_matrix, 0)
    
    return weighted_adjacency_matrix



#
def GetRepr(X, basis=None):
    """
    Compute the representation of a symmetric matrix X in a specific basis.
    """
    d = X.shape[0]
    if basis is None:
        g = d * (d + 1) // 2
        y = np.zeros(g)
        k = 0
        for i in range(d):
            for j in range(i + 1):
                if i == j:
                    y[k] = X[i, i]
                else:
                    y[k] = (X[i, j] + X[j, i]) / sqrt(2)
                k += 1
    else:
        # If a basis is provided, use it to calculate the representation
        g = len(basis)
        y = np.zeros(g)
        for i in range(g):
            y[i] = np.sum(X * basis[i])
    
    return y


def Reconstruct(y, basis=None):
    """
    Reconstruct the matrix from its representation y.
    """
    g = len(y)
    d = int(np.floor(sqrt(2 * g)))
    X = np.zeros((d, d))

    if basis is None:
        k = 0
        for i in range(d):
            for j in range(i + 1):
                if i == j:
                    X[i, i] = y[k]
                else:
                    X[i, j] = y[k] / sqrt(2)
                    X[j, i] = y[k] / sqrt(2)
                k += 1
    else:
        for i in range(g):
            X += y[i] * basis[i]
    
    return X




# ---------- Compute representation for variables used for computing of the asymptotci distribution ----- 


def ComputeProjection(Q, S, X, Y):
    """
    Compute the scalar product <dT^S_{Q}(X), Y>.
    
    Parameters:
    - Q: A covariance matrix.
    - S: Another covariance matrix.
    - X: A matrix to project.
    - Y: A matrix to project.
    
    Returns:
    - The scalar product result.
    """
    sqrt_S = sqrtm1(S)                     # S^{1/2}
    R = sqrt_S @ Q @ sqrt_S               # S^{1/2} Q S^{1/2}
    eigvals, eigvecs = eigh(R)  # U^* Lambda U
    
    Delta = eigvecs.T @ sqrt_S @ X @ sqrt_S @ eigvecs  # Delta = U S^{1/2} X S^{1/2} U^*
    d = len(eigvals)
    
    Delta = Delta / (sqrt(eigvals)[:, None] + sqrt(eigvals)[None, :])
    
    inv_Lambda = diag(1 / sqrt(eigvals))  # Lambda^{-1/2}
    dT = -sqrt_S @ eigvecs @ inv_Lambda @ Delta @ inv_Lambda @ eigvecs.T @ sqrt_S
    
    return np.trace(dT @ Y)

def ComputeReprdT(Q, S=None, basis=None):
    """
    Compute the differential map dT^{S}_{Q}(X).
    
    Parameters:
    - Q: A covariance matrix.
    - S: Another covariance matrix. Defaults to Q if not provided.
    - basis: Optional basis to project onto.
    
    Returns:
    - The differential map represented in the chosen basis.
    """
    if S is None:
        S = Q
    
    sqrt_S = sqrtm1(S)                     # S^{1/2}
    R = sqrt_S @ Q @ sqrt_S               # S^{1/2} Q_* S^{1/2}
    eigvals, eigvecs = scipy.linalg.eigh(R)  # U^* Lambda U
    
    sqrt_S = eigvecs.T @ sqrt_S
    d = len(eigvals)
    
    if basis is None:
        g = d * (d + 1) // 2
        Deltas = []
        for i in range(d):
            for j in range(i + 1):
                if i == j:
                    Deltas.append(np.outer(sqrt_S[:, i], sqrt_S[:, i]))
                else:
                    Deltas.append((np.outer(sqrt_S[:, i], sqrt_S[:, j]) +
                                   np.outer(sqrt_S[:, j], sqrt_S[:, i])) / sqrt(2))
    else:
        g = len(basis)
        Deltas = [sqrt_S @ basis[i]['u'] @ sqrt_S.T for i in range(g)]
    
    q = sqrt(np.maximum(eigvals, 1e-7))
    r = np.ones(d)
    dummy = 1/(np.outer(q, r) + np.outer(r, q)) 
    dummy /= (np.outer(q, r)*np.outer(r, q))
#     dummy = np.nan_to_num(dummy, nan=0)
#     dummy = 1 / (sqrt(eigvals)[:, None] + sqrt(eigvals)[None, :])
#     dummy /= (sqrt(eigvals)[:, None] * sqrt(eigvals)[None, :])  
    repr_map = np.zeros((g, g))
    for i in range(g):
        A = -Deltas[i] * dummy
        for j in range(g):
            repr_map[i, j] = np.sum(A * Deltas[j])
    
    return repr_map


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

def GetOTmap(Q, S):
    """
    Generate the optimal transport map from N(0, Q) to N(0, S).

    Parameters:
    - Q: Covariance matrix of the first distribution.
    - S: Covariance matrix of the second distribution.

    Returns:
    - OT: The optimal transport map matrix.
    """
    # Step 1: Compute SqrtmInv(Q)
    D = sqrtmInv2(Q)
    
    # Step 2: Compute Sqrtm(Q)
    K = sqrtm1(Q)
    
    # Step 3: Compute Sqrtm(K @ S @ K)
    G = sqrtm1(K @ S @ K)
    
    # Step 4: Compute the optimal transport map OT
    OT = D @ G @ D
    
    return OT


def GetRepr(X, basis=None):
    """
    Compute the representation of a matrix X in the specified basis.
    
    Parameters:
    - X: The matrix to represent.
    - basis: Optional basis for projection. If None, default symmetric matrix basis is used.
    
    Returns:
    - Representation vector y.
    """
    d = X.shape[0]
    
    if basis is None:
        g = d * (d + 1) // 2
        y = np.zeros(g)
        k = 0
        for i in range(d):
            for j in range(i + 1):
                if i == j:
                    y[k] = X[i, i]
                else:
                    y[k] = (X[i, j] + X[j, i]) / sqrt(2)
                k += 1
    else:
        g = len(basis)
        y = np.zeros(g)
        for i in range(g):
            y[i] = np.sum(X * basis[i]['u'])
    
    return y

def Reconstruct(y, basis=None):
    """
    Reconstruct the matrix from its vector representation y.
    
    Parameters:
    - y: The vector representation of the matrix.
    - basis: Optional basis for reconstruction. If None, use the default symmetric matrix basis.
    
    Returns:
    - The reconstructed matrix X.
    """
    g = len(y)
    d = int(np.floor(sqrt(2 * g)))  # Calculate the dimension of the matrix
    X = np.zeros((d, d))  # Initialize the output matrix with zeros
    
    if basis is None:
        k = 0
        for i in range(d):
            for j in range(i + 1):  # Only iterate over the upper triangle, including the diagonal
                k += 1
                if i == j:
                    X[i, i] = y[k - 1]  # Diagonal elements
                else:
                    X[i, j] = y[k - 1] / sqrt(2)
                    X[j, i] = y[k - 1] / sqrt(2)  # Symmetric element
    else:
        for i in range(g):
            X += y[i] * basis[i]['u']  # Use the provided basis for reconstruction
    
    return X





#------------------------- Aux functions -----------------------

