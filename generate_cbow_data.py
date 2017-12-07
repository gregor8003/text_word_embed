import gc
import sys

import numpy as np
import pandas as pd

from utils import (
    get_mdsd_csv_indexed_file,
    get_mdsd_csv_cbow_data_input_file,
    get_mdsd_csv_cbow_data_output_file,
    get_vocabulary_size,
)
from utils_training_data import generate_training_data_cbow


CBOW_WINDOW_HALF_SIZE = 3
CBOW_TRAINING_NEGATIVES = 30


def main():

    main_path = sys.argv[1]

    try:
        cbow_window_half_size = sys.argv[2]
    except KeyError:
        cbow_window_half_size = CBOW_WINDOW_HALF_SIZE

    try:
        cbow_training_negatives = sys.argv[3]
    except KeyError:
        cbow_training_negatives = CBOW_TRAINING_NEGATIVES

    minimum_review_length = 2 * cbow_window_half_size + 1

    mdsd_csv_indexed_file = get_mdsd_csv_indexed_file(main_path=main_path)
    mdsd_csv_cbow_data_input_file = (
        get_mdsd_csv_cbow_data_input_file(main_path=main_path)
    )
    mdsd_csv_cbow_data_output_file = (
        get_mdsd_csv_cbow_data_output_file(main_path=main_path)
    )

    print('Reading', mdsd_csv_indexed_file, '... ', end='', flush=True)
    df = pd.read_csv(mdsd_csv_indexed_file, encoding='utf-8')
    print('done')

    cbow_input = []
    cbow_output = []
    k = 1

    max_index = get_vocabulary_size(main_path=main_path)

    for rowtuple in df.itertuples(index=True, name=None):
        index, indexed_text, _, _, _ = rowtuple
        indexed_text = [int(s) for s in indexed_text.split()]

        if len(indexed_text) < minimum_review_length:
            continue

        input_npy, output_npy = generate_training_data_cbow(
            indexed_text, cbow_window_half_size, max_index,
            negatives_cnt=cbow_training_negatives
        )

        cbow_input.append(input_npy)
        cbow_output.append(output_npy)

        if index > 0:
            if index % 10 == 0:
                print('.', end='', flush=True)
            if index % 100 == 0:
                print(index, 'of', df.shape[0])

        if len(cbow_input) == 1000:
            cbow_input = np.concatenate(cbow_input, axis=0)
            cbow_output = np.concatenate(cbow_output, axis=0)

            input_path = mdsd_csv_cbow_data_input_file.format(k)
            print('Writing', input_path)
            np.save(input_path, cbow_input)
            print('Done')

            output_path = mdsd_csv_cbow_data_output_file.format(k)
            print('Writing', output_path)
            np.save(output_path, cbow_output)
            print('Done')

            cbow_input = []
            cbow_output = []
            k = k + 1

            gc.collect()


if __name__ == '__main__':
    main()
