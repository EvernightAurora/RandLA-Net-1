import numpy as np
from os.path import join, split, exists
import os
import sys
from tqdm import tqdm
sys.path.append('..')
from stddef import Test_Setting_Path, Final_Path


test_set = ['50', '51']
output_path = join(Final_Path, 'SD_Result')
os.mkdir(output_path) if not exists(output_path)
log_file_name = join(output_path, 'mylog.out')
logf = open(log_file_name, 'at')


def log_out(str):
    logf.write(str)
    print(str)
    logf.flush()


def get_file_id(f):
    a = split(f)
    b = split(a[0])
    return b[1]


def sdc_main():
    file_list = []
    leader_path = join(Test_Setting_Path, 'Pred_SD', 'Raw0')
    for fn in os.listdir(leader_path):
        if fn in test_set:
            n_path = join(leader_path, fn)
            for f in os.listdir(n_path):
                if f[-5:] == '.prob':
                    file_list.append(join(n_path, f))
    var_sum = np.zeros((12,), dtype=np.float32)
    var_square = np.zeros_like(var_sum)
    var_count = np.zeros_like(var_sum)
    for f in tqdm(file_list):
        probs = []
        for i in range(5):
            l_path = join(Test_Setting_Path, 'Pred_SD', 'Raw{}'.format(i))
            f_path = join(l_path, get_file_id(f), split(f)[1])
            probs.append(np.fromfile(f_path, dtype=np.float32))
        probs = np.array(probs)
        probs_mean = np.mean(probs, axis=0)                     # E
        probs_square = np.square(probs)
        probs_square_sum = np.sum(probs_square, axis=2)         # ff^t
        probs_mean_square_sum = np.sum(np.square(probs_mean), axis=1)       # EE^t
        var = np.mean(probs_square_sum, axis=0) - probs_mean_square_sum
        var = var.astype(np.float32)
        os.mkdir(join(output_path, get_file_id(f))) if not exists(join(output_path, get_file_id(f)))
        var.tofile(join(output_path, get_file_id(f), split(f)[1][:-5] + '.conf'))
        label = np.argmax(probs_mean, axis=1) + 1
        label.tofile(join(output_path, get_file_id(f), split(f)[1][:-5] + '.label'))

        # extension
        for lab in range(12):
            pos = np.where(label == lab)
            part_var = var[pos]
            var_sum[lab] += np.sum(part_var, axis=0)
            var_count[lab] += len(pos)
            var_square[lab] += np.sum(np.square(part_var), axis=0)

    print('Finished')
    s0 = s1 = s2 = ''
    s0 = 'mean\t\t'
    s1 = 'var\t\t'
    s2 = 'cnt\t\t'
    var_mean = var_sum / var_count
    var_var = var_square / var_count - np.square(var_mean)
    for i in range(12):
        s0 += '\t{}\t'.format(var_mean[i])
        s1 += '\t{}\t'.format(var_var[i])
        s2 += '\t{}\t'.format(var_count)
    log_out(s0)
    log_out(s1)
    log_out(s2)



