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
    if row:
        return fea / norms
    else:
        return (fea.T / norms).T

def zsl_el(S_est, S_te_gt, hitk, testclasses_id, test_labels):
    # INPUT:
    # % S_est: estimated semantic labels
    # % S_te_gt: ground truth semantic labels
    # % param: other parameters

    dist = 1 - scipy.spatial.distance.cosine(S_est, NormalizeFea(S_te_gt, 0))
    Y_hit5 = np.zeros(dist.shape[0], hitk)
    n = 0
    for i in range(dist.shape[0]):
        if Y_hit5[i,:] in test_labels:
            n += 1

    return n / dist.shape[0], Y_hit5