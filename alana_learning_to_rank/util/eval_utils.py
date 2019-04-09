import numpy as np


def compute_hits(in_true, in_pred):
    hits = []
    for true, pred_list in zip(in_true, in_pred):
        hits.append(int(true == pred_list[0]))
    return np.mean(hits)


def get_prec_recall(tp, fp, fn):
    precision = tp / (tp + fp + 10e-20)
    recall = tp / (tp + fn + 10e-20)
    f1 = 2 * precision * recall / (precision + recall + 1e-20)
    return precision, recall, f1


def get_tp_fp_fn(label_list, pred_list):
    tp = len([t for t in pred_list if t in label_list])
    fp = max(0, len(pred_list) - tp)
    fn = max(0, len(label_list) - tp)
    return tp, fp, fn


def compute_f1(in_true, in_pred):
    f1s = []
    for true, pred_list in zip(in_true, in_pred):
        _, _, f1 = get_prec_recall(get_tp_fp_fn(true, pred_list[0]))
        f1s.append(f1)
    return np.mean(f1s)


def eval_accuracy(in_pred_true, in_pred_fake):
    accuracy = sum(map(lambda x: 0 < x, in_pred_true - in_pred_fake)) / float(len(in_pred_true))
    return accuracy[0]
