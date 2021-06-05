import sys, os, time
import logging
import numpy as np
logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False

from conlleval import evaluate as con_eval

num_labels = {
    'cola': 2,
    'sst-2': 2,
    'chns': 2,
    'cged': 9,
    'fce': 6,
}

class CgedProcessor(object):
    def get_labels(self):
        return ['O', 'B-R', 'I-R', 'B-M', 'I-M', 'B-S', 'I-S', 'B-W', 'I-W']

class FceProcessor(object):
    def get_labels(self):
        return ['O', 'B-R', 'I-R', 'B-U', 'I-U', 'B-M']

task_processors = {
    'cged': CgedProcessor,
    'fce' : FceProcessor,
}

def is_sklearn_available():
    return _has_sklearn

if _has_sklearn:

    def simple_accuracy(preds, labels):
        return {'acc': (preds == labels).mean()}


    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }


    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

def evaluate(task, eval_data):
    if task == 'cola':
        preds = np.array(eval_data['preds'])
        labels = np.array(eval_data['labels'])
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task == 'chns' or task == 'sst-2':
        preds = np.array(eval_data['preds'])
        labels = np.array(eval_data['labels'])
        return simple_accuracy(preds, labels)
    elif task == 'fce':
        preds = eval_data['preds']
        labels = eval_data['labels']
        loss_mask = eval_data['loss_mask']
        true_tags, pred_tags = [], []
        label_list = FceProcessor().get_labels()
        for i in range(len(preds)):
            for pid, tid, mask in zip(preds[i], labels[i], loss_mask[i]):
                if mask == 1:
                    pred_tags.append(label_list[pid])
                    true_tags.append(label_list[tid])
            pred_tags.append('O')
            true_tags.append('O')
        prec, rec, f1 = con_eval(true_tags, pred_tags, verbose=False)
        return {
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'f0.5': 1.25 * prec * rec / (rec + 0.25 * prec)
        }
    else:
        raise KeyError(task)