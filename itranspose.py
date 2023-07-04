def itranspose(a, axes):
    iaxes = sorted(list(range(0, len(axes))), key=lambda x: axes[x])
    return a.transpose(iaxes)