import functools
import sys

from keras import metrics
from keras.layers import Dense, Embedding, Flatten
from keras.models import Sequential
import numpy as np

from generate_cbow_data import (
    CBOW_TRAINING_NEGATIVES,
    CBOW_WINDOW_HALF_SIZE,
)
from utils import (
    get_mdsd_cbow_embedding_weights_file,
    get_vocabulary_size,
)
from utils_batch import (
    closest_batch_size,
    fit_generator,
    get_mdsd_csv_cbow_data_input_size,
)


EMBED_SIZE = 300


def main():

    main_path = sys.argv[1]

    try:
        embed_size = sys.argv[2]
    except KeyError:
        embed_size = EMBED_SIZE

    try:
        cbow_window_half_size = sys.argv[3]
    except KeyError:
        cbow_window_half_size = CBOW_WINDOW_HALF_SIZE

    try:
        cbow_training_negatives = sys.argv[4]
    except KeyError:
        cbow_training_negatives = CBOW_TRAINING_NEGATIVES

    total_size = get_mdsd_csv_cbow_data_input_size(main_path=main_path)

    base_factor = cbow_training_negatives + 2 * cbow_window_half_size
    seed_size = base_factor * 1000
    batch_size = closest_batch_size(total_size, seed_size)

    max_index = get_vocabulary_size(main_path=main_path)

    nn = Sequential()

    l1 = Embedding(
        input_dim=max_index + 1,
        output_dim=embed_size,
        input_length=2
    )
    nn.add(l1)

    l2 = Flatten()
    nn.add(l2)

    l3 = Dense(units=2, activation='sigmoid')
    nn.add(l3)

    nn.compile(
        optimizer='adagrad',
        loss='binary_crossentropy',
        metrics=[metrics.MSE, metrics.MSLE, metrics.MAE, metrics.MAPE],
    )

    print(nn.summary())

    fit_generator_gen = functools.partial(fit_generator, batch_size)()

    steps_per_epoch = total_size // batch_size

    history = nn.fit_generator(
        generator=fit_generator_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=1,
        verbose=1
    )

    print(history.history)

    embedding_weights = l1.get_weights()

    mdsd_cbow_embedding_weights_file = get_mdsd_cbow_embedding_weights_file(
        main_path=main_path
    )

    print('Writing', mdsd_cbow_embedding_weights_file)
    np.save(mdsd_cbow_embedding_weights_file, embedding_weights)
    print('Done')


if __name__ == '__main__':
    main()
