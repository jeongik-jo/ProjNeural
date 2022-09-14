import FcNeural
#import KernelRegression
import KNN
import ProjNeural
import RBF
import time
from scipy.stats import iqr
import numpy as np


repeat_time = 50
repeat_func = ProjNeural.main


def main():
    start = time.time()
    losses = []
    for i in range(repeat_time):
        print('repeat %d' % i)
        losses.append(repeat_func(i))
        print()

    print('\nmedian:\t', np.median(losses))
    print('iqr:\t', iqr(losses))
    print('total time:\t', time.time() - start)

main()
