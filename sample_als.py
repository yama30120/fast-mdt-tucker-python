import numpy as np
from als import als
from all_prod import all_prod

A = np.random.rand(10, 5)
U, s, Vt = np.linalg.svd(A)
U = U[:, 0:5]
S = np.diag(s)
V = Vt.T

print('A')
print(A)

print('U @ S @ V.T')
print(U @ S @ V.T)

R = [1, 1]
G, F = als(A, R)
X = all_prod(G, F)
print('G x {U}')
print(X)

print('ans')
print(U[:, 0:R[0]] @ S[0:R[0], 0:R[1]] @ V[:, 0:R[1]].T)
