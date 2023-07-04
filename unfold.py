# input:
# X: a N-th order tensor (size: (X_0, X_1, ..., X_(N-1))) (ndarray in numpy module)
# n: a mode
# 
# output:
# mode-k unfolded matrix (X_n, X_0 x ... x X_(n-1) x X_(n+1) x ... x X_(N-1))
def unfold(X, k, order='C'):
    N = X.ndim
    perm = [k] + list(range(0, k)) + list(range(k + 1, N))
    return X.transpose(perm).reshape(X.shape[k], -1, order=order)