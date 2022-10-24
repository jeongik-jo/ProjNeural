import FcNeural
import KernelRegression
import KNN
import RBF
import MARS
import ProjNeural
import time
from scipy.stats import iqr
import numpy as np
import csv
import Dataset
import os


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
    if not os.path.exists('results'):
        os.makedirs('results')

    if Dataset.is_m1:
        f_name = 'm1'
    elif Dataset.is_m2:
        f_name = 'm2'
    elif Dataset.is_m3:
        f_name = 'm3'
    elif Dataset.is_m4:
        f_name = 'm4'
    elif Dataset.is_m5:
        f_name = 'm5'

    with open('results/' + f_name + '_' + 
              str(Dataset.train_data_size + Dataset.test_data_size) + '_samples_' +
              str(Dataset.noise_strength) + '_noise_' +
              method_name + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['median', 'iqr', 'mean', 'stddev', 'total time'])
        writer.writerow([np.median(losses), iqr(losses), np.mean(losses), np.std(losses, ddof=1), time.time() - start])


if __name__ == "__main__":
    main()
