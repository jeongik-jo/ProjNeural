import FcNeural
import KernelRegression
import KNN
import ProjNeural
import RBF
import MARS
import time
from scipy.stats import iqr
import numpy as np


repeat_time = 50
repeat_func = FcNeural.main
method_name = 'FcNeural' + str(FcNeural.depth)


def main():
    start = time.time()
    losses = []
    for i in range(repeat_time):
        print('repeat %d' % i)
        losses.append(repeat_func(i))
        print()

    with open(method_name + '.txt', 'w') as f:
        f.write('\nmedian:\t' + str(np.median(losses)))
        f.write('\niqr:\t' + str(iqr(losses)))
        f.write('\nmean:\t' + str(np.mean(losses)))
        f.write('\nstddev:\t' + str(np.std(losses, ddof=1)))
        f.write('\ntotal time:\t' + str(time.time() - start))

if __name__ == "__main__":
    main()
