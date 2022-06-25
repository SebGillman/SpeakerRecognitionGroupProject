import distutils.util

import numpy as np
from tqdm import tqdm


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' default: %(default)s.',
                           **kwargs)


#  use cosine similarity to calculate accuracy and best threshold
def cal_accuracy_threshold(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_accuracy = 0
    best_threshold = 0
    for i in tqdm(range(0, 100)):
        threshold = i * 0.01
        y_test = (y_score >= threshold)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold

    return best_accuracy, best_threshold


# cal accuracy
def cal_accuracy(y_score, y_true, threshold=0.5):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    y_test = (y_score >= threshold)
    accuracy = np.mean((y_test == y_true).astype(int))
    return accuracy


# cos similarity
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

