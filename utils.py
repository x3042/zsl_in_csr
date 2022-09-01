import numpy as np
import scipy

def corrupt(X, rate):
    m, n = X.shape
    rate = n * rate // 100
    ind_m = np.random.randint(m, size=rate)
    ind_n = np.random.randint(n, size=rate)

    for i, j in zip(ind_m, ind_n):
        X[i][j] = 0

    return X

def NormalizeFea(fea, row=1):
# % if row == 1, normalize each row of fea to have unit norm;
# % if row == 0, normalize each column of fea to have unit norm;

    norms = np.linalg.norm(fea, axis=row)
    if row == 0:
        return fea / norms
    else:
        return (fea.T / norms).T


def distCosine(x, y):
    xx = np.sum(x**2, axis=1)**0.5
    x = x / xx[:, np.newaxis]
    yy = np.sum(y**2, axis=1)**0.5
    y = y / yy[:, np.newaxis]
    dist = 1 - np.dot(x, y.transpose())
    return dist


def zsl_el(S_est, S_te_gt, hitk, testclasses_id, test_labels):
    # INPUT:
    # % S_est: estimated semantic labels
    # % S_te_gt: ground truth semantic labels
    # % param: other parameters

    dist = 1 - distCosine(S_est, NormalizeFea(S_te_gt, 0))
    Y_hit5 = np.zeros((dist.shape[0], hitk))
    for i in range(dist.shape[0]):
        sorted_id = sorted(range(len(dist[i, :])), key=lambda k: dist[i, :][k], reverse=True)
        Y_hit5[i, :] = testclasses_id[sorted_id[0:hitk]]

    n = 0
    for i in range(dist.shape[0]):
        if test_labels[i] in Y_hit5[i, :]:
            n += 1

    return n / dist.shape[0], Y_hit5
