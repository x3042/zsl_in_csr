from utils import NormalizeFea, corrupt, zsl_el
from scipy import linalg
import numpy as np
from scipy.io import loadmat

awa = loadmat('/Users/demitrakill/Downloads/awa_demo_data_v7.mat')
X_tr = awa['X_tr']
X_test = awa['X_te']
S_tr = awa['S_tr']
test_labels = awa['test_labels']
test_classes_id = awa['testclasses_id']
S_te_gt = awa['S_te_gt']
S_te_pro = awa['S_te_pro']
X_tr = NormalizeFea(X_tr, 0)

X = X_tr.T
S = S_tr.T

X_c = corrupt(X.T, 5).T
X_c = NormalizeFea(X_c)

lam = 6 * 10**5
beta = 8 * 10**11
rank = 36

A = S @ S.T
B = lam * X @ X_c.T  # in original there is B = lambda * X_c*X_c'; but in the article B=lamXX^.T; X^ - corrupted
C = S @ X.T + lam * S @ X_c.T
k, _ = S.shape
U = 0

for i in range(5):

    W = linalg.solve_sylvester(A, B, C)
    V, _, _ = np.linalg.svd(W @ W.T, full_matrices=True)
    U = np.eye(k) - V[:, :rank+1] @ V[:, :rank+1].T
    A = S @ S.T + beta * U

    hitk = 1
    testclasses_id = awa["testclasses_id"];
    test_labels = awa['test_labels']
    S_est = NormalizeFea(X_test) @ NormalizeFea(W).T

    zsl_accuracy, Y_hit5 = zsl_el(S_est, S_te_gt, hitk, testclasses_id, test_labels)
    print('ZSL accuracy V >> S', zsl_accuracy*100)

    X_te_pro = NormalizeFea(S_te_pro, 0) @ NormalizeFea(W)
    zsl_accuracy, _ = zsl_el(X_test, X_te_pro, hitk, testclasses_id, test_labels)

    print('ZSL accuracy S >> V', zsl_accuracy * 100)
