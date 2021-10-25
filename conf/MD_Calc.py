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
learn_set = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
output_path = join(Final_Path, 'MD_Result')
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


def md_learn():
    log_out("Now learning")
    file_list = []
    learn_path = join(Test_Setting_Path, 'Pred_OM')
    for fn in os.listdir(learn_path):
        if fn in learn_set:
            npath = join(learn_path, fn)
            for f in os.listdir(npath):
                if f[-5:] == '.prob':
                    file_list.append(join(npath, f))
    learn_sum = np.zeros((12, 11), dtype=np.float64)
    learn_num = np.zeros_like(learn_sum)
    learn_square = np.zeros_like(learn_sum)

    for f in tqdm(file_list):
        prob = np.fromfile(f, dtype=np.float32)
        prob = prob.reshape((-1, 11))
        true_label = np.fromfile(join(true_label_path, get_file_id(f), split(f)[1][:-5] + '.t_label'),
                                 dtype=np.uint32)
        for lab in range(12):
            pos = np.where(true_label == lab)
            part_prob = prob[pos]
            learn_sum[lab] += np.sum(part_prob, axis=0)
            learn_num[lab] += len(pos[0])
            learn_square[lab] += np.sum(np.square(part_prob), axis=0)
    learn_mean = learn_set / learn_num
    learn_var = np.sum(learn_square / learn_num, axis=1) - np.sum(np.square(learn_mean), axis=1)
    learn_var = np.sum(learn_var, axis=0)
    np.save(join(output_path, 'Mean.npy'), learn_mean)
    np.save(join(output_path, 'Var.npy'), learn_var)
    log_out('----------------------learning result------------------------')
    for i in range(12):
        log_out(repr(learn_mean[i]))
    log_out('Var = {}'.format(learn_var))
    log_out('----------------------------End------------------------------')


def load_learn_result():
    mean = np.load(join(output_path, 'Mean.npy'))
    var = np.load(join(output_path, 'Var.npy'))
    return mean, var



def md_main():
    md_learn()
    l_mean, l_var = load_learn_result()
    log_out('calculating')
    file_list = []
    leader_path = join(Test_Setting_Path, 'Pred_OM')
    for fn in os.listdir(leader_path):
        if fn in test_set:
            n_path = join(leader_path, fn)
            for f in os.listdir(n_path):
                if f[-5:] == '.prob':
                    file_list.append(join(n_path, f))
    score_sum = np.zeros((12,), dtype=np.float64)
    score_count = np.zeros_like(score_sum)
    score_square = np.zeros_like(score_sum)
    print('summary {} Files'.format(len(file_list)))
    pred_trans = np.zeros((12, 12), dtype=np.uint32)

    for f in tqdm(file_list):
        prob = np.fromfile(f, dtype=np.float32)
        prob = prob.reshape((-1, 11))
        f_pred = np.fromfile(join(Test_Setting_Path, 'Pred_OM', get_file_id(f), split(f)[1][:-5] + 'label'),
                             dtype=np.uint32)

        score = np.zeros((prob.shape[0], 12), dtype=np.float32)

        for lab in range(12):
            prob_lab = prob - l_mean
            prob_lab_square = np.sum(np.square(prob_lab), axis=1)
            prob_m = - prob_lab_square * np.reciprocal(l_var)
            score[:, lab] = prob_m
        pred = np.argmax(score, axis=1)
        m_score = np.max(score, axis=1)
        for i in range(pred.shape[0]):
            pred_trans[f_pred[i]][pred[i]] += 1

        os.mkdir(join(output_path, get_file_id(f))) if not exists(join(output_path, get_file_id(f))) else None

        pred.tofile(join(output_path, get_file_id(f), split(f)[1][:-5] + '.label'))
        m_score.tofile(join(output_path, get_file_id(f), split(f)[1][:-5] + '.conf'))
        f_pred.tofile(join(output_path, get_file_id(f), split(f)[1][:-5] + '.f_label'))

        # extension
        true_label = np.fromfile(join(true_label_path, get_file_id(f), split(f)[1][:-5] + '.t_label'), dtype=np.uint32)
        for lab in range(12):
            pos = np.where(true_label == lab)
            part_score = m_score[pos]
            score_sum[lab] += np.sum(part_score, axis=0)
            score_count[lab] += len(pos[0])
            score_square[lab] += np.sum(np.square(part_scale), axis=0)

    print('Finished')
    s0 = s1 = s2 = ''
    s0 = 'mean\t'
    s1 = 'var\t'
    s2 = 'cnt\t'
    score_mean = score_sum / score_count
    score_var = score_square / score_count - np.square(score_mean)
    for i in range(12):
        s0 += '%e\t' % (score_mean[i])
        s1 += '%e\t' % (score_var[i])
        s2 += '%d\t' % (int(score_count[i]))
    log_out(s0)
    log_out(s1)
    log_out(s2)


if __name__ == '__main__':
    md_main()