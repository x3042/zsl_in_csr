import numpy as np

def NormalizeFea(fea, row):
# % if row == 1, normalize each row of fea to have unit norm;
# % if row == 0, normalize each column of fea to have unit norm;

    if row is None:
        row = 1

    norms = np.linalg.norm(fea, axis=row)
    if row:
        return fea / norms
    else:
        return fea.T / norms
