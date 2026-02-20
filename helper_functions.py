import numpy as np  
#import math

def LU(matrix): #function that takes a matrix and gives P,L,U as output, or error if matrix is singular
    A=matrix.copy()
    A=A.astype(float)
    n=len(A)
    P=np.eye(n) # Identity matrix of size n
    L=np.eye(n)
    for i in range(n):
        max_col_relative=np.argmax(A[i:n, i]) # index of max val in column with "i" indexed as "0"
        max_col=max_col_relative+i # index of max val in the column, lower than i 
        pivot=A[max_col,i] 
        if np.isclose(pivot, 0.0): # instead of checking pivot=0.0, checking within epsilon of 0
            print("Error: Singular")
            return 
        A[[i,max_col]]=A[[max_col,i]] #Swapping rows of A
        P[[i,max_col]]=P[[max_col,i]] #Rows swapped in P : same as multiplying P_i to P     
        for j in range(i+1,n): # Converting elements of all elements under pivot to zero
            mul= A[j,i]/pivot
            A[j,i:]=A[j,i:]-(mul*A[i,i:])
            L[j,i]=mul # Changing the elements of L to respective multipliers, same as multiplying L_i ^-1 to L
        if i>0:
                L[[i,max_col], :i] = L[[max_col,i], :i] #Swapping columns upto "i-1" : same as P_i^-1 * L_k * P_i^-1, k<i
    return P,L,A

def solve_LU(P,L,U, b):
    n = len(U)
    # Apply the permutation to b to match the row swaps
    Pb = np.dot(P, b)
    
    # Forward Substitution (Ly=Pb)
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])
    
    # Backward Substitution(Ux=y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        
    return x

def GenerateOfSize(x):
    n=x
# Using standard_normal to get real-valued (float) numbers 
    rng = np.random.default_rng(seed=42) #Setting seed for reproducibility
    m = rng.standard_normal((n, n)) 
    vec = rng.standard_normal(n)
    return m,vec


