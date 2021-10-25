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


T = 1000.0

test_set = ['50', '51']
output_path = join(Final_Path, 'ODIN_Result')
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


def odinc_main():
    file_list = []
    leader_path = join(Test_Setting_Path, 'Pred_OM')
    for fn in os.listdir(leader_path):
        if fn in test_set:
            n_path = join(leader_path, fn)
            for f in os.listdir(n_path):
                if f[-5:] == '.prob':
                    file_list.append(join(n_path, f))
    scale_sum = np.zeros((12,), dtype=np.float64)
    scale_square = np.zeros_like(scale_sum)
    scale_count = np.zeros_like(scale_sum)
    print('summary {} Files'.format(len(file_list)))
    for f in tqdm(file_list):
        probs = np.fromfile(f, dtype=np.float32)
        probs = np.reshape(probs, (-1, 11))
        temped_probs = probs / T
        exp_temped_probs = np.exp(temped_probs)
        pred = np.argmax(exp_temped_probs, axis=1).astype(np.uint32)
        pred = pred + 1
        
        max_exp_probs = np.max(exp_temped_probs, axis=1)
        sum_exp_probs = np.sum(exp_temped_probs, axis=1)
        scale = max_exp_probs / sum_exp_probs
        
        os.mkdir(join(output_path, get_file_id(f))) if not exists(join(output_path, get_file_id(f))) else None
        
        pred.tofile(join(output_path, get_file_id(f), split(f)[1][:-5] + '.label'))
        scale.tofile(join(output_path, get_file_id(f), split(f)[1][:-5] + '.conf'))

        # extension
        true_label = np.fromfile(join(true_label_path, get_file_id(f), split(f)[1][:-5] + '.t_label'), dtype=np.uint32)
        for lab in range(12):
            pos = np.where(true_label == lab)
            part_scale = scale[pos]
            scale_sum[lab] += np.sum(part_scale, axis=0)
            scale_count[lab] += len(pos[0])
            scale_square[lab] += np.sum(np.square(part_scale), axis=0)

    print('Finished')
    s0 = s1 = s2 = ''
    s0 = 'mean\t'
    s1 = 'var\t'
    s2 = 'cnt\t'
    scale_mean = scale_sum / scale_count
    scale_var = scale_square / scale_count - np.square(scale_mean)
    for i in range(12):
        s0 += '%e\t'%(scale_mean[i])
        s1 += '%e\t'%(scale_var[i])
        s2 += '%d\t'%(int(scale_count[i]))
    log_out(s0)
    log_out(s1)
    log_out(s2)


if __name__ == '__main__':
    odinc_main()