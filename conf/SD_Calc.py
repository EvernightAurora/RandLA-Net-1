import numpy as np
from os.path import join, split, exists, dirname, abspath
import os
import sys
from tqdm import tqdm
BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from stddef import Test_Setting_Path, Final_Path


test_set = ['50', '51']
output_path = join(Final_Path, 'SD_Result')
os.mkdir(output_path) if not exists(output_path) else None
log_file_name = join(output_path, 'mylog.txt')
logf = open(log_file_name, 'at')
true_label_path = r'/home/pyc/xf/RLN-Train/Pred_OM/'


def log_out(str):
    str += '\r\n'
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
    var_sum = np.zeros((12,), dtype=np.float64)
    var_square = np.zeros_like(var_sum)
    var_count = np.zeros_like(var_sum)
    print('summary {} Files'.format(len(file_list)))
    for f in tqdm(file_list):
        probs = []
        # print(f)
        for i in range(5):
            l_path = join(Test_Setting_Path, 'Pred_SD', 'Raw{}'.format(i))
            f_path = join(l_path, get_file_id(f), split(f)[1])
            probs.append(np.reshape(np.fromfile(f_path, dtype=np.float32), (-1,11)))
            # print(probs[-1].shape)
        probs = np.array(probs)
        
        probs_mean = np.mean(probs, axis=0)                     # E
        probs_square = np.square(probs)
        probs_square_sum = np.sum(probs_square, axis=2)         # ff^t
        probs_mean_square_sum = np.sum(np.square(probs_mean), axis=1)       # EE^t
        var = np.mean(probs_square_sum, axis=0) - probs_mean_square_sum
        var = var.astype(np.float32)
        os.mkdir(join(output_path, get_file_id(f))) if not exists(join(output_path, get_file_id(f))) else None
        var.tofile(join(output_path, get_file_id(f), split(f)[1][:-5] + '.conf'))
        label = np.argmax(probs_mean, axis=1) + 1
        label.tofile(join(output_path, get_file_id(f), split(f)[1][:-5] + '.label'))

        # extension
        true_label = np.fromfile(join(true_label_path, get_file_id(f), split(f)[1][:-5] + '.t_label'), dtype=np.uint32)
        for lab in range(12):
            pos = np.where(true_label == lab)
            part_var = var[pos]
            var_sum[lab] += np.sum(part_var, axis=0)
            var_count[lab] += len(pos[0])
            var_square[lab] += np.sum(np.square(part_var), axis=0)

    print('Finished')
    s0 = s1 = s2 = ''
    s0 = 'mean\t'
    s1 = 'var\t'
    s2 = 'cnt\t'
    var_mean = var_sum / var_count
    var_var = var_square / var_count - np.square(var_mean)
    for i in range(12):
        s0 += '\t%10f\t'%(var_mean[i])
        s1 += '\t%10f\t'%(var_var[i])
        s2 += '\t%d\t'%(int(var_count[i]))
    log_out(s0)
    log_out(s1)
    log_out(s2)


if __name__ == '__main__':
    sdc_main()