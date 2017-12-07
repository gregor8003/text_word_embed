import csv
import sys

from scipy.spatial import cKDTree
import numpy as np

from utils import (
    get_mdsd_cbow_embedding_weights_file,
    get_mdsd_cbow_wordvec_closest_neighbors_file,
    get_mdsd_cbow_wordvec_closest_neighbors_csv_file,
    read_mdsd_index2word_pck_file,
    INDEX_UNKNOWN_WORD,
    WORD_UNKNOWN_WORD,
)


NEAREST_NEIGHBORS_CNT = 100

KDTREE_LEAFSIZE = 16

INDEX2WORD_REMAINING = {
    INDEX_UNKNOWN_WORD: WORD_UNKNOWN_WORD,
}


def find_word(index2word, word_index):
    try:
        word = index2word[word_index]
    except KeyError:
        word = INDEX2WORD_REMAINING[word_index]
    return word


def main():

    main_path = sys.argv[1]

    try:
        nearest_neighbors_cnt = sys.argv[2]
    except KeyError:
        nearest_neighbors_cnt = NEAREST_NEIGHBORS_CNT

    try:
        kdtree_leafsize = sys.argv[3]
    except KeyError:
        kdtree_leafsize = KDTREE_LEAFSIZE

    print('Loading word vectors')
    mdsd_cbow_embedding_weights_file = get_mdsd_cbow_embedding_weights_file(
        main_path=main_path
    )
    word_vectors = np.load(mdsd_cbow_embedding_weights_file)
    print(word_vectors.shape)
    print('done')

    print('Flatten word vector matrix')
    # reshape all but last dimension
    word_vectors = word_vectors.reshape(-1, word_vectors.shape[-1])
    print(word_vectors.shape)
    print('done')

    nwords = word_vectors.shape[0]

    print('Build query tree of word vectors')
    words_tree = cKDTree(data=word_vectors, leafsize=kdtree_leafsize)
    print('done')

    nearest_neighbors = []

    print(
        'Finding', nearest_neighbors_cnt, 'nearest neighbors for each of',
        nwords, 'words'
    )
    for index in range(nwords):
        word_vector = word_vectors[index]
        _, neighbors = words_tree.query(
            [word_vector], k=nearest_neighbors_cnt + 1, eps=0, p=2,
            distance_upper_bound=np.inf, n_jobs=1
        )
        # as query tree contains all points, the closest returned point
        # is equal to query point
        # remove first element of [1,n] array and reduce to (n,)
        neighbors = np.delete(neighbors, [0], axis=1)[0]
        nearest_neighbors.append(neighbors)

        if index > 0:
            if index % 10 == 0:
                print('.', end='', flush=True)
            if index % 100 == 0:
                print(index, 'of', nwords)

    nearest_neighbors = np.stack(nearest_neighbors, axis=0)
    print(nearest_neighbors.shape)
    print('done')

    mdsd_cbow_wordvec_closest_neighbors_file = (
        get_mdsd_cbow_wordvec_closest_neighbors_file(main_path=main_path)
    )
    print('Writing', mdsd_cbow_wordvec_closest_neighbors_file)
    np.save(
        mdsd_cbow_wordvec_closest_neighbors_file, nearest_neighbors
    )
    print('done')

    print('Loading index2word')
    index2word = read_mdsd_index2word_pck_file(main_path=main_path)
    print('done')

    mdsd_cbow_wordvec_closest_neighbors_csv_file = (
        get_mdsd_cbow_wordvec_closest_neighbors_csv_file(main_path=main_path)
    )
    print('Writing', mdsd_cbow_wordvec_closest_neighbors_csv_file)
    with open(mdsd_cbow_wordvec_closest_neighbors_csv_file,
              'wt', encoding='utf-8', newline='') as wf:
        csvw = csv.writer(wf)
        header = [
            'Nearest Word %d' % nn for nn in range(1, nearest_neighbors_cnt+1)
        ]
        header.insert(0, 'Word')
        csvw.writerow(header)
        # word index 0 is for unknown word, skip it
        for word_index, nns in enumerate(nearest_neighbors[1:], 1):
            word = find_word(index2word, word_index)
            row = [word]
            for nn in nns:
                word_nn = find_word(index2word, nn)
                row.append(word_nn)
            csvw.writerow(row)

            if word_index > 0:
                if word_index % 10 == 0:
                    print('.', end='', flush=True)
                if word_index % 100 == 0:
                    print(word_index, 'of', nwords)

    print()
    print('done')


if __name__ == '__main__':
    main()
