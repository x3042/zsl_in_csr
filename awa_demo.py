from utils import NormalizeFea, corrupt, zsl_el
import scipy
import numpy as np

X_tr = NormalizeFea(X_tr, 0)

X = X_tr
S = S_tr

X_c = corrupt(X, 5)
X_c = NormalizeFea(X_c, 0)

lam = 6 * 10**5
beta = 8 * 10**11
rank = 36

A = S * S.T
B = lam * X * X_c.T
C = S * X.T + lam * S * X_c.T
k, _ = S.shape
U = 0

for i in range(5):

    W = scipy.linalg.solve_sylvester(A, B, C)
    V, _, _ = np.linalg.svd(W*W.T, full_matrices=True)
    U = np.eye(k) - V[, :rank+1] * V[, :rank+1].T
    A = S * S.T + beta * U

    hitk = 1
    testclasses_id = testclasses_id;
    test_labels = test_labels
    S_est = NormalizeFea(X_test) * NormalizeFea(W).T

    zsl_accuracy, Y_hit5 = zsl_el(S_est, S_te_gt, hitk, testclasses_id, test_labels)
    print('ZSL accuracy V >> S', zsl_accuracy*100)

    X_te_pro = NormalizeFea(S_te_pro, 0) * NormalizeFea(W)
    zsl_accuracy, _ = zsl_el(X_test, X_te_pro, hitk, testclasses_id, test_labels)
    print('ZSL accuracy S >> V', zsl_accuracy * 100)