from bisect import bisect_left
import itertools

from numpy.lib.npyio import load as npload
import numpy as np

from utils import (
    get_mdsd_csv_cbow_data_input_files_iter,
    get_mdsd_csv_cbow_data_output_files_iter,
    grouper,
    MDSD_MAIN_PATH,
)


def closest_batch_size(total_size, seed_size):
    # produce all factors of total_size
    factors = [d for d in range(1, total_size // 2 + 1) if not total_size % d]
    # select number closest to seed size
    # https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
    pos = bisect_left(factors, seed_size)
    if pos == 0:
        return factors[0]
    if pos == len(factors):
        return factors[-1]
    before = factors[pos - 1]
    after = factors[pos]
    closest = before
    if after - seed_size < seed_size - before:
        closest = after
    return closest


def gen_array(array):
    for els in np.nditer(array, flags=['external_loop']):
        for el in grouper(els, 2):
            yield el


def get_mdsd_csv_cbow_data_input_iter(batch_size, main_path=MDSD_MAIN_PATH):
    iterators = []
    for path in get_mdsd_csv_cbow_data_input_files_iter(main_path=main_path):
        data = npload(path, mmap_mode='r')
        iterators.append(gen_array(data))
    main_it = itertools.chain.from_iterable(iterators)
    for batch in grouper(main_it, batch_size):
        batch = np.stack(batch, axis=0)
        yield batch


def get_mdsd_csv_cbow_data_output_iter(batch_size, main_path=MDSD_MAIN_PATH):
    iterators = []
    for path in get_mdsd_csv_cbow_data_output_files_iter(main_path=main_path):
        data = npload(path, mmap_mode='r')
        iterators.append(gen_array(data))
    main_it = itertools.chain.from_iterable(iterators)
    for batch in grouper(main_it, batch_size):
        batch = np.stack(batch, axis=0)
        yield batch


def fit_generator(batch_size, main_path=MDSD_MAIN_PATH):
    while True:
        for res in fit_generator_it(batch_size, main_path):
            yield res


def fit_generator_it(batch_size, main_path=MDSD_MAIN_PATH):
    gens = zip(
        get_mdsd_csv_cbow_data_input_iter(batch_size, main_path=main_path),
        get_mdsd_csv_cbow_data_output_iter(batch_size, main_path=main_path)
    )
    for tr, lb in gens:
        yield tr, lb


def get_mdsd_csv_cbow_data_input_size(main_path=MDSD_MAIN_PATH):
    total_size = 0
    for path in get_mdsd_csv_cbow_data_input_files_iter(main_path=main_path):
        data = npload(path, mmap_mode='r')
        total_size = total_size + data.shape[0]
    return total_size


def get_mdsd_csv_cbow_data_output_size(main_path=MDSD_MAIN_PATH):
    total_size = 0
    for path in get_mdsd_csv_cbow_data_output_files_iter(main_path=main_path):
        data = npload(path, mmap_mode='r')
        total_size = total_size + data.shape[0]
    return total_size
