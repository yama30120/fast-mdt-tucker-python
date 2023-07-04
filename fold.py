def fold(Xk, k, dim, order='C'):
    N = len(dim)
    perm = [k] + list(range(0, k)) + list(range(k + 1, N))
    dim_perm = []
    for i in perm:
        dim_perm.append(dim[i])
    
    iperm = list(range(1, k + 1)) + [0] + list(range(k + 1, N))
    return Xk.reshape(dim_perm, order=order).transpose(iperm)