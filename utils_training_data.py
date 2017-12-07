import gc
import itertools

from sklearn.preprocessing import normalize
import numpy as np

from utils import grouper


DEFAULT_NEGATIVES = 30


def get_running_window_middlepoint_data(iterable, window_half_size):
    mp_datas = []
    iter_size = len(iterable)
    window_size = 2 * window_half_size + 1
    left_end = window_half_size
    right_end = window_half_size + 1
    middle_point = window_half_size
    for i in range(iter_size - window_size + 1):
        iter_part = iterable[i:i + window_size]
        mp_data = (
            iter_part[:left_end],
            iter_part[middle_point],
            iter_part[right_end:]
        )
        mp_datas.append(mp_data)
    return mp_datas


def _generate_probs(max_index, positives):
    # construct proper probability distribution by excluding positives
    p_values = [1] * (max_index + 1)
    p_values[0] = 0
    for p in positives:
        p_values[p] = 0
    p_values_norm = normalize([p_values], norm='l1')[0]
    return p_values_norm


def generate_negatives(max_index, ngroups, positives, negatives_cnt):
    total_negatives_cnt = ngroups * negatives_cnt
    all_values = np.arange(max_index + 1)
    p_values = _generate_probs(max_index, positives)
    negatives = np.random.choice(
        all_values, total_negatives_cnt, replace=False, p=p_values
    )
    return list(grouper(negatives, negatives_cnt))


def generate_training_data_cbow(iterable, window_half_size, max_index,
                                negatives_cnt=DEFAULT_NEGATIVES):

    input_npy = []
    output_npy = []

    labels_positive = (1, 1)
    labels_negative = (1, 0)

    mp_datas = get_running_window_middlepoint_data(iterable, window_half_size)
    all_negatives = generate_negatives(
        max_index, len(mp_datas), iterable, negatives_cnt
    )
    for mp_data, negatives in zip(mp_datas, all_negatives):
        middle = mp_data[1]
        context_words = itertools.chain(mp_data[0], mp_data[2])
        for context_word in context_words:
            input_npy.append([middle, context_word])
            output_npy.append(labels_positive)
        for negative in negatives:
            input_npy.append([middle, negative])
            output_npy.append(labels_negative)

    input_npy = np.array(input_npy)
    output_npy = np.array(output_npy)

    gc.collect()

    return input_npy, output_npy
