import FcNeural
import KernelRegression
import KNN
import RBF
import MARS
import ProjNeural
import time
from scipy.stats import iqr
import numpy as np


is_fcneural = True
if is_fcneural:
    depth = 1
is_kernel = False
is_knn = False
is_mars = False
is_projneural = False
is_rbf = False

repeat_time = 50
if is_fcneural:
    repeat_func = FcNeural.main
    method_name = 'FcNeural' + str(depth)
elif is_kernel:
    repeat_func = KernelRegression.main
    method_name = 'Kernel'
elif is_knn:
    repeat_func = KNN.main
    method_name = 'KNN'
elif is_mars:
    repeat_func = MARS.main
    method_name = 'MARS'
elif is_projneural:
    repeat_func = ProjNeural.main
    method_name = 'ProjNeural'
elif is_rbf:
    repeat_func = RBF.main
    method_name = 'RBF'
else:
    raise AssertionError


def main():
    start = time.time()
    losses = []
    for i in range(repeat_time):
        print('repeat %d' % i)
        if is_fcneural:
            loss = repeat_func(i, depth)
        else:
            loss = repeat_func(i)
        losses.append(loss)
        print()

    with open(method_name + '.txt', 'w') as f:
        f.write('\nmedian:\t' + str(np.median(losses)))
        f.write('\niqr:\t' + str(iqr(losses)))
        f.write('\nmean:\t' + str(np.mean(losses)))
        f.write('\nstddev:\t' + str(np.std(losses, ddof=1)))
        f.write('\ntotal time:\t' + str(time.time() - start))

if __name__ == "__main__":
    main()
